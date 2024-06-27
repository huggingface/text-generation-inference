import torch
import time

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models import Model
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch

from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.utils.dist import MEMORY_FRACTION

tracer = trace.get_tracer(__name__)

from transformers.cache_utils import PagedCache

from loguru import logger

# Why define it here?
BLOCK_SIZE: int = 16


class CausalLMRagged(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # TODO felix: fix support for accelerate
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()

        self.kv_cache = []
        self.num_layers = len(model.model.layers)
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_size = model.config.hidden_size // model.config.num_attention_heads

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        super().__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    def warmup(self, batch: FlashCausalLMBatch):
        # The warmup batch is the biggest batch we could ever receive
        empty_cache()

        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.dtype,
                self.device,
            )
            max_bt = batch.max_blocks
            max_s = max_bt * BLOCK_SIZE

            _, batch, _ = self.generate_token(batch)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            ) from e

        synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        free_memory = get_free_memory(self.device, MEMORY_FRACTION)
        batch_num_blocks = batch.num_blocks if batch is not None else 0

        num_blocks = (
            # Leave 5% for some wiggle room
            int((free_memory * 0.95) // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        del batch

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
            self.device,
        )

        return int(num_blocks * BLOCK_SIZE)

    def init_kv_cache(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.kv_cache = []
        empty_cache()

        element_size = torch.tensor([], dtype=dtype).element_size()
        if SYSTEM == "ipex" and device.type == "xpu":
            raise ValueError("Untested. Please open an issue")
        else:
            x = BLOCK_SIZE // element_size

        if SYSTEM == "ipex" and device == torch.device("cpu"):
            raise ValueError("Untested. Please open an issue")

        self.kv_cache = [
            (
                torch.empty(
                    (num_blocks, num_heads, head_size // x, BLOCK_SIZE, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, head_size, BLOCK_SIZE),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(num_layers)
        ]

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def forward(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: adapter_data: not supported

        input_ids = batch.input_ids
        position_ids = batch.position_ids
        cu_seqlen_prefill = batch.cu_seqlen_prefill
        kv_cache = self.kv_cache
        block_tables = batch.block_tables_tensor
        slots = batch.slots[batch.slot_indices]
        input_lengths = batch.input_lengths_tensor
        max_s = batch.max_seqlen
        lm_head_indices = batch.prefill_head_indices

        # TODO felix: support window attention
        # if cu_seqlen_prefill is None and self.max_past() is not None:
        #     # In decode, not prefill, we're actually overwriting the KV-cache
        #     # in a circular buffer mode.
        #     # This makes sure the max_s for the decode pass is correct.
        #     max_s = min(self.max_past(), max_s)

        bs = input_ids.shape[0]

        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=PagedCache(),
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            prefill_cache_indices=batch.prefill_cache_indices,
            lm_head_indices=lm_head_indices,
            cache_position=False,
            return_dict=False,
        )[0]

        if lm_head_indices is not None:
            logits = logits[lm_head_indices]

        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None

        speculative_logits = None

        return logits, speculative_logits

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        prefill = batch.cu_seqlen_prefill is not None
        prefill_logprobs = batch.prefill_next_token_indices is not None

        # Update adapter indices for speculative tokens (if present)
        # adapter_meta = batch.adapter_meta
        # if batch.speculative_ids is not None:
        #     B, speculative_length = batch.speculative_ids.shape
        #     new_length = speculative_length + 1
        #     adapter_indices = (
        #         adapter_meta.adapter_indices.unsqueeze(-1)
        #         .expand(B, new_length)
        #         .reshape(-1)
        #     )
        #     adapter_segments = adapter_meta.adapter_segments * new_length
        #     adapter_meta = AdapterBatchMetadata(
        #         adapter_indices=adapter_indices,
        #         adapter_set=adapter_meta.adapter_set,
        #         adapter_segments=adapter_segments,
        #         segment_indices=adapter_meta.segment_indices,
        #     )

        # Assign pointers to adapter weights
        # TODO(travis): don't update this if indices haven't changed
        # adapter_data = AdapterBatchData.from_meta(
        #     adapter_meta,
        #     self.layer_to_adapter_weights,
        #     prefill,
        #     batch.prefill_head_indices,
        # )

        logger.info(f"batch.input_ids {batch.input_ids}")
        out, speculative_logits = self.forward(batch)

        logger.info(f"out {out.shape}")
        logger.info(f"speculative_logits {speculative_logits}")

        if prefill:
            next_token_logits = (
                out[batch.prefill_next_token_indices] if prefill_logprobs else out
            )
            if speculative_logits is not None:
                speculative_logits = (
                    speculative_logits[batch.prefill_next_token_indices]
                    if prefill_logprobs
                    else speculative_logits
                )
            # next_adapter_indices = batch.adapter_meta.adapter_indices.new_empty(
            #     len(batch)
            # )

        else:
            next_token_logits = out
            # next_adapter_indices = batch.adapter_meta.adapter_indices

        speculate = get_speculate()
        (
            next_input_ids,
            next_token_logprobs,
            logprobs,
            accepted_ids,
            speculative_ids,
        ) = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_seqlen],
            next_token_logits,
            speculate,
            batch.speculative_ids,
            speculative_logits,
        )

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens, batch.top_n_tokens_tensor, logprobs, accepted_ids
        )

        if prefill:
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))

            next_position_ids = batch.position_ids.new_empty(len(batch))
            batch.slot_indices = batch.slot_indices[batch.cu_seqlen_prefill[1:] - 1]
            # We do not need cu_seqlen_prefill anymore
            batch.cu_seqlen_prefill = None
        else:
            prefill_logprobs = None
            next_position_ids = batch.position_ids

        # Cumulative length
        cumulative_length = 0

        # Results
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(batch.input_lengths, batch.all_input_ids, accepted_ids)

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        index = 0
        for i, (input_length, all_input_ids, n_accepted_ids) in enumerate(iterator):
            # Indexing metadata
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            if prefill:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]
                out_length = out_end_index - out_start_index

                # Initialize position_ids
                # In decode, we do not need this as we can just increment position ids
                next_position_ids[i] = batch.position_ids[end_index - 1]

                # Initialize adapter indices
                # In decode, we only have one token per row in the batch, so grab last index
                # next_adapter_indices[i] = batch.adapter_meta.adapter_indices[
                #     end_index - 1
                # ]

                # Used to gather prefill logprobs
                # Copy batch.input_ids to prefill_token_indices
                if prefill_logprobs:
                    if len(batch) > 1:
                        prefill_tokens_indices[out_start_index : out_end_index - 1] = (
                            batch.input_ids[start_index + 1 : start_index + out_length]
                        )
                    else:
                        # Set prefill_tokens_indices to the correct slice
                        prefill_tokens_indices = batch.input_ids[
                            start_index + 1 : start_index + out_length
                        ]

            for j in range(n_accepted_ids):
                batch.all_input_ids_tensor[i, input_length + j] = next_input_ids[index]
                index += 1

            cumulative_length += input_length

        logger.info(f"batch.input_lengths_tensor {batch.input_lengths_tensor}")
        logger.info(f"accepted_ids {accepted_ids}")
        logger.info(f"batch.all_input_ids {batch.all_input_ids}")

        # Update values
        batch.input_ids = next_input_ids[accepted_ids.cumsum(dim=-1) - 1]
        batch.speculative_ids = speculative_ids
        batch.position_ids = next_position_ids + accepted_ids
        batch.input_lengths_tensor += accepted_ids
        batch.slot_indices += accepted_ids
        # batch.adapter_meta.adapter_indices = None

        # if prefill:
        #     # adjust segment lengths to account for all request lengths being 1 during decoding
        #     adapter_segments, _ = find_segments(batch.adapter_meta.adapter_indices)
        #     batch.adapter_meta.adapter_segments = torch.tensor(
        #         adapter_segments,
        #         dtype=torch.int32,
        #         device=batch.adapter_meta.adapter_segments.device,
        #     )

        if prefill and prefill_logprobs:
            # Get prefill logprobs
            prefill_logprobs_tensor = torch.log_softmax(out, -1)
            prefill_logprobs = torch.gather(
                prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1)
            )
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids.tolist()
        accepted_ids = accepted_ids.tolist()
        start_decode = time.time_ns()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            batch.top_n_tokens,
            accepted_ids,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # For each member of the batch
        index = 0
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            do_sample,
            seed,
            top_n_tokens,
            n_accepted_ids,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Append next token to all tokens
            next_token_texts = []
            left = 0

            if n_accepted_ids > 1:
                if RANK == 0:
                    logger.debug(f"Speculated ids {n_accepted_ids - 1}")

            current_stopped = False
            for j in range(index, index + n_accepted_ids):
                # Generated token
                next_token_id = next_token_ids[j]
                all_input_ids.append(next_token_id)
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids,
                    prefix_offset,
                    read_offset,
                )
                next_token_texts.append(next_token_text)

                stop, reason = stopping_criteria(
                    next_token_id,
                    next_token_text,
                )

                if stop:
                    left = index + n_accepted_ids - j - 1
                    current_stopped = True
                    break
                else:
                    current_stopped = False
            stopped = stopped and current_stopped

            _next_token_ids = next_token_ids[index : index + n_accepted_ids - left]
            _next_token_logprobs = next_token_logprobs[
                index : index + n_accepted_ids - left
            ]
            index += n_accepted_ids

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids,
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if prefill and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]

                    # Remove generated token to only have prefill and add nan for first prompt token
                    request_prefill_logprobs = [float("nan")] + prefill_logprobs[
                        out_start_index : out_end_index - 1
                    ]
                    prefill_token_ids = all_input_ids[:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )

                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        request_prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        _next_token_ids,
                        _next_token_logprobs,
                        next_token_texts,
                        [nid in self.all_special_ids for nid in _next_token_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # accept each new token for this specific request since we may
            # have more than one new token per request with speculative decoding
            for next_token_id in _next_token_ids:
                batch.next_token_chooser = (
                    batch.next_token_chooser.advance_grammar_single(i, next_token_id)
                )

            # Update values
            batch.input_lengths[i] = input_length + n_accepted_ids
            if batch.input_lengths[i] > batch.max_seqlen:
                batch.max_seqlen = batch.input_lengths[i]
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            # No need to return a batch if we know that all requests stopped
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        batch.prefill_cu_outlens = None
        batch.prefill_head_indices = None
        batch.prefill_next_token_indices = None

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)
