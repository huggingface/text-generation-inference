import torch
import torch.distributed
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional
from text_generation_server.models.custom_modeling.mamba_modeling import (
    MambaConfig,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
import time
from text_generation_server.models.custom_modeling.mamba_modeling import MambaModel
from text_generation_server.models import Model
from typing import Any, List, Optional, Tuple, Type, Dict
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils.tokens import batch_top_tokens, Sampling
from dataclasses import dataclass
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from mamba_ssm.utils.generation import InferenceParams


@dataclass
class MambaBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Metadata used for padding
    max_input_length: int
    padding_right_offset: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    # Past metadata
    keys_head_dim_last: bool = True

    # Inference params
    inference_params: Optional[Dict[str, Any]] = None

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "MambaBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        prefix_offsets = []
        read_offsets = []
        requests_idx_mapping = {}

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0
        for i, r in enumerate(pb.requests):
            requests_idx_mapping[r.id] = i
            inputs.append(r.inputs)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            max_truncation = max(max_truncation, r.truncate)
            max_decode_tokens += stopping_criteria.max_new_tokens
            padding_right_offset = max(
                padding_right_offset, stopping_criteria.max_new_tokens
            )

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        ).to(device)
        for _ in pb.requests:
            input_len = tokenized_inputs["input_ids"].shape[1]
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()
        input_ids = tokenized_inputs["input_ids"]
        all_input_ids = tokenized_inputs["input_ids"].T.split(1, dim=1)
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )
        max_tokens = len(inputs) * (max_input_length + max_decode_tokens)
        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            # past_input_ids=None,
            all_input_ids=list(all_input_ids),
            input_lengths=input_lengths.tolist(),
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length.item(),
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
        )

    def filter(self, request_ids: List[int]) -> Optional["MambaBatch"]:
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        max_input_length = 0

        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []

        total_remaining_decode_tokens = 0
        new_padding_right_offset = 0

        indices = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)

            requests.append(self.requests[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])
            all_input_ids.append(self.all_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)
            indices.append(idx)

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(self.top_n_tokens[idx])
            remaining_decode_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )
            total_remaining_decode_tokens += remaining_decode_tokens
            new_padding_right_offset = max(
                new_padding_right_offset, remaining_decode_tokens
            )

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        input_ids = self.input_ids[keep_indices]

        top_n_tokens_tensor = self.top_n_tokens_tensor[keep_indices]
        max_tokens = len(request_ids) * max_input_length + total_remaining_decode_tokens

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = input_ids
        self.all_input_ids = all_input_ids
        self.input_lengths = input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        self.next_token_choosers = next_token_choosers
        self.stopping_criterias = stopping_criterias
        self.top_n_tokens = top_n_tokens
        self.top_n_tokens_tensor = top_n_tokens_tensor
        self.max_input_length = max_input_length
        self.padding_right_offset = new_padding_right_offset
        self.max_tokens = max_tokens

        # TODO
        # Kept it simple by just updating the state, maybe updating the other CPU values is necessary.
        key_value_memory_dict = {}
        for i, (
            conv_state,
            ssm_state,
        ) in self.inference_params.key_value_memory_dict.items():
            key_value_memory_dict[i] = (conv_state[indices], ssm_state[indices])
        self.inference_params.key_value_memory_dict = key_value_memory_dict

        return self

    @classmethod
    def concatenate(cls, batches: List["MambaBatch"]) -> "MambaBatch":
        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        max_tokens = 0
        max_seqlen = 0
        batch_size = 0
        seqlen_offset = 0

        # Batch tensors
        input_ids = None
        top_n_tokens_tensor = None

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)
            top_n_tokens.extend(batch.top_n_tokens)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            if top_n_tokens_tensor is None:
                top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
                    total_batch_size,
                )
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (
                max_input_length - batch.max_input_length
            ) * len(batch)

            max_seqlen = max(max_seqlen, batch.inference_params.max_seqlen)
            seqlen_offset = max(seqlen_offset, batch.inference_params.seqlen_offset)
            batch_size += batch.inference_params.max_batch_size

            start_index = end_index

        (_, d_model, d_conv) = (
            batches[0].inference_params.key_value_memory_dict[0][0].shape
        )
        (_, _, d_state) = batches[0].inference_params.key_value_memory_dict[0][1].shape
        n_blocks = len(batches[0].inference_params.key_value_memory_dict)
        dtype = batches[0].inference_params.key_value_memory_dict[0][0].dtype
        device = batches[0].inference_params.key_value_memory_dict[0][0].device

        key_value_memory_dict = {}
        for i in range(n_blocks):
            conv_state = torch.zeros(
                batch_size,
                d_model,
                d_conv,
                device=device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                d_model,
                d_state,
                device=device,
                dtype=dtype,
            )
            key_value_memory_dict[i] = (conv_state, ssm_state)
        lengths_per_sample = torch.zeros(batch_size, dtype=torch.int32, device=device)

        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_offset,
            key_value_memory_dict=key_value_memory_dict,
            lengths_per_sample=lengths_per_sample,
        )

        current_batch = 0
        for batch in batches:
            for i in range(n_blocks):
                conv_state, ssm_state = batch.inference_params.key_value_memory_dict[i]
                batch_size = batch.inference_params.max_batch_size
                inference_params.key_value_memory_dict[i][0][
                    current_batch : current_batch + batch_size
                ] = conv_state
                inference_params.key_value_memory_dict[i][1][
                    current_batch : current_batch + batch_size
                ] = ssm_state
                inference_params.lengths_per_sample[
                    current_batch : current_batch + batch_size
                ] = batch.inference_params.lengths_per_sample
            current_batch += batch_size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
            max_tokens=max_tokens,
            inference_params=inference_params,
        )

    def __len__(self):
        return len(self.requests)


class Mamba(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, _rank, _world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        config = MambaConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        config.quantize = quantize
        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        model = MambaModel(config, weights)
        torch.distributed.barrier(group=self.process_group)
        super(Mamba, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[MambaBatch]:
        return MambaBatch

    def warmup(self, batch) -> Optional[int]:
        # TODO: implement warmup for Mamba if needed
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        past: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            past=past,
        )

    def generate_token(self, batch) -> Tuple[List[Any], Optional[Any], Tuple[int, int]]:
        start = time.time_ns()
        input_ids = (
            batch.input_ids
        )  # batch.past_input_ids if batch.past_input_ids is not None else batch.input_ids

        batch_size = input_ids.shape[0]
        max_seqlen = input_ids.shape[1]
        dtype = input_ids.dtype

        # Inference params
        seqlen_og = 0
        inf_cache = {}
        lengths_per_sample = (
            torch.ones(batch_size, dtype=torch.int32, device=input_ids.device)
            * max_seqlen
        )

        if batch.inference_params is None:
            inference_params = InferenceParams(
                max_seqlen=max_seqlen,
                max_batch_size=batch_size,
                seqlen_offset=seqlen_og,
                key_value_memory_dict=inf_cache,
                lengths_per_sample=lengths_per_sample,
            )

            # Allocate inference cache
            for res_block in self.model.blocks:
                block = res_block.mamba_block
                conv_state = torch.zeros(
                    batch_size,
                    self.model.config.d_model * self.model.config.expand,
                    self.model.config.d_conv,
                    device=block.conv1d.weight.device,
                    dtype=block.conv1d.weight.dtype,
                )
                ssm_state = torch.zeros(
                    batch_size,
                    self.model.config.d_model * self.model.config.expand,
                    self.model.config.d_state,
                    device=block.dt_proj.weight.device,
                    dtype=block.dt_proj.weight.dtype,
                )
                inference_params.key_value_memory_dict[block.layer_idx] = (
                    conv_state,
                    ssm_state,
                )
            batch.inference_params = inference_params

        # Forward pass
        logits, past_input_ids, new_inference_params = self.model(
            input_ids, batch.inference_params
        )

        batch.inference_params = new_inference_params
        # Results
        generations: List[Generation] = []
        stopped = True

        # Speculation is not active for causal
        accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1], -1),
            accepted_ids,
        )

        start_decode = time.time_ns()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids[:, 0],
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )
                else:
                    generated_text = None

                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + torch.log_softmax(
                        logits, -1
                    ).gather(1, all_input_ids[1:]).squeeze(1)[
                        -new_input_length:-1
                    ].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    toptoken_texts = self.tokenizer.batch_decode(
                        top_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    special_toptokens = [
                        token_id in self.all_special_ids for token_id in top_token_ids
                    ]
                    top_tokens = Tokens(
                        top_token_ids,
                        top_token_logprobs,
                        toptoken_texts,
                        special_toptokens,
                    )
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id_squeezed],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id_squeezed.item() in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

                # Update values
                batch.input_ids[i, 0] = next_token_id
                batch.all_input_ids[i] = all_input_ids
                batch.input_lengths[i] = new_input_length
                batch.prefix_offsets[i] = prefix_offset
                batch.read_offsets[i] = read_offset
                batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)
