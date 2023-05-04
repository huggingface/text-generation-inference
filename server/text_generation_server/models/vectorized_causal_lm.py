import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict, Union
from loguru import logger

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import StoppingCriteria

tracer = trace.get_tracer(__name__)


@dataclass
class VectorizedCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]]]

    # All tokens
    input_ids: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    offsets: List[Optional[int]]
    token_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: "VectorizedNextTokenChooser"
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    max_input_length: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    kv_cache_seq_dim:int=2

    def to_pb(self) -> generate_pb2.Batch:
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=len(self),
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "VectorizedCausalLMBatch":
        inputs = [r.inputs for r in pb.requests]
        offsets = [None]*len(inputs)
        token_offsets = [None]*len(inputs)
        requests_idx_mapping = {r.id:i for i, r in enumerate(pb.requests)}

        # Parse batch
        stopping_criterias = [StoppingCriteria.from_pb(r.stopping_parameters, tokenizer) for r in pb.requests]
        max_new_tokens=(stopping_criteria.max_new_tokens for stopping_criteria in stopping_criterias)

        next_token_chooser=VectorizedNextTokenChooser.from_pb([r.parameters for r in pb.requests], device)

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max(r.truncate for r in pb.requests),
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max().item()

        input_shape=(pb.size, max_input_length + max(max_new_tokens))

        # Allocate maximum attention_mask
        attention_mask = torch.empty(input_shape, dtype=torch.bool, device=device)
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length].copy_(tokenized_inputs["attention_mask"])
        attention_mask[:, max_input_length:].fill_(1)

        position_ids = attention_mask.cumsum(-1).sub_(1)
        position_ids[:, :max_input_length].relu_()

        input_ids = torch.empty(input_shape, dtype=torch.int64, device=device)
        input_ids[:, :max_input_length].copy_(tokenized_inputs["input_ids"])

        max_tokens = len(inputs) * max_input_length + sum(max_new_tokens)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            input_ids=input_ids,
            input_lengths=input_lengths.tolist(),
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length.item(),
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, requests: List[generate_pb2.Request]) -> Optional["VectorizedCausalLMBatch"]:
        if len(requests) == 0:
            raise ValueError("Batch must have at least one request")
        if len(requests) == len(self):
            return self

        self.requests = requests
        keep_indices = [self.requests_idx_mapping[r.id] for r in self.requests]

        # New values after filtering
        self.requests_idx_mapping={r.id:i for i, r in enumerate(self.requests)}
        self.input_lengths=[self.input_lengths[i] for i in keep_indices]
        self.offsets = [self.offsets[i] for i in keep_indices]
        self.token_offsets = [self.token_offsets[i] for i in keep_indices]
        self.next_token_chooser=self.next_token_chooser.filter(keep_indices)
        self.stopping_criterias = [self.stopping_criterias[i] for i in keep_indices]
        remaining_decode_tokens=[stopping_criteria.max_new_tokens - stopping_criteria.current_tokens for stopping_criteria in self.stopping_criterias]

        # Select the remaining indices and remove unnecessary padding
        max_input_length=max(self.input_lengths)
        sequence_slice=slice(self.max_input_length-max_input_length, self.max_input_length+max(remaining_decode_tokens))
        self.max_input_length=max_input_length
        self.max_tokens = len(self.requests) * self.max_input_length + sum(remaining_decode_tokens)

        self.input_ids = self.input_ids[keep_indices,sequence_slice]
        self.position_ids = self.position_ids[keep_indices,sequence_slice]
        self.attention_mask = self.attention_mask[keep_indices,sequence_slice]

        tensors_to_update = []
        if self.past_key_values is not None:
            if not isinstance(self.past_key_values,(list, tuple)):
                raise NotImplementedError(f"Unsupported kv cache type: {type(self.past_key_values)}")
            for layer_kv in self.past_key_values:
                if isinstance(layer_kv, torch.Tensor):
                    tensors_to_update.append(layer_kv)
                elif isinstance(layer_kv,(list, tuple)):
                    tensors_to_update.extend(layer_kv)
                else:
                    raise NotImplementedError(f"Unsupported layer  kv cache type: {type(layer_kv)}")

        kv_cache_slice=[keep_indices, *(slice(None) for _ in range(1, self.kv_cache_seq_dim)), sequence_slice]
        for tensor in tensors_to_update:
            # Update tensors in-place to allow incremental garbage collection
            tensors_to_update.data=tensor[kv_cache_slice]

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["VectorizedCausalLMBatch"]) -> "VectorizedCausalLMBatch":
        if len(batches)==0:
            raise ValueError("Cannot concatenate empty list.")
        requests=[request for batch in batches for request in batch.requests]
        batch_sizes=[len(batch.requests) for batch in batches]
        batch_size=sum(batch_sizes)

        end_indices=torch.tensor(batch_sizes).cumsum(0).tolist()
        start_indices=[0]+end_indices[:-1]

        input_lengths = [length for batch in batches for length in batch.input_lengths]
        offsets = [offset for batch in batches for offset in batch.offsets]
        token_offsets = [token_offset for batch in batches for token_offset in batch.token_offsets]
        next_token_chooser=VectorizedNextTokenChooser.concatenate([batch.next_token_chooser for batch in batches])
        stopping_criterias = [stopping_criteria for batch in batches for stopping_criteria in batch.stopping_criterias]

        requests_idx_mapping = {k: v + start_index for batch, start_index in zip(batches, start_indices) for k, v in batch.requests_idx_mapping.items()}

        max_input_length=max(input_lengths)
        left_indices=[max_input_length-batch.max_input_length for batch in batches]

        input_shape=(batch_size, max_input_length + max(batch.input_ids.size(1)-batch.max_input_length for batch in batches))
        device=batches[0].input_ids.device

        # Allocate maximum attention_mask
        attention_mask = torch.empty(input_shape, dtype=torch.bool, device=device)
        attention_mask[:, :max_input_length].fill_(0)
        attention_mask[:, max_input_length:].fill_(1)

        input_ids = torch.empty(input_shape, dtype=torch.int64, device=device)
        # TODO : only needed for prefill
        input_ids[:, :max_input_length].fill_(0)

        for batch,start_index, end_index, left_index in zip(batches, start_indices, end_indices, left_indices):
            attention_mask[start_index:end_index, left_index:max_input_length].copy_(batch.attention_mask[:, :batch.max_input_length])
            input_ids[start_index:end_index, left_index:max_input_length].copy_(batch.input_ids[:, :batch.max_input_length])

        position_ids = attention_mask.cumsum(-1).sub_(1)
        position_ids[:, :max_input_length].relu_()

        max_tokens = sum(batch.max_tokens + (max_input_length - batch.max_input_length) * len(batch) for batch in batches)

        kv_formats=None
        for batch in batches:
            if batch.past_key_values is None:
                raise ValueError("Only concatenate prefilled batches")
            if not isinstance(batch.past_key_values, (list, tuple)):
                raise NotImplementedError(f"Unsupported kv cache type: {type(batch.past_key_values)}")
            if kv_formats is None:
                num_layers=len(batch.past_key_values)
                if num_layers==0:
                    raise ValueError("Empty KV cache")
                kv_formats = [0]*num_layers
            elif len(batch.past_key_values)!=len(kv_formats):
                raise ValueError("Num layers is not constant")
            for i, layer_kv in enumerate(batch.past_key_values):
                if isinstance(layer_kv, (list, tuple)):
                    kv_format = len(layer_kv)
                else:
                    kv_format=None
                if kv_formats[i]==0:
                    if kv_format==0:
                        raise ValueError("Empty KV cache")
                    kv_formats[i]=kv_format
                elif kv_formats[i]!=kv_format:
                    raise ValueError("Incompatible KV cache format.")

        kv_cache_seq_dim=batches[0].kv_cache_seq_dim
        past_key_values=[]
        for i, kv_format in enumerate(kv_formats):
            for j in range(1 if kv_format is None else kv_format):
                tensors_to_merge=[batch.past_key_values[i] for batch in batches]
                # Generally `max_input_length`, unless the model allocates more than needed.
                right_indices=[left_index+tensor.size(kv_cache_seq_dim) for tensor, left_index in zip(tensors_to_merge, left_indices)]
                combined_shape=[batch_size]+list(tensors_to_merge[0].shape[1:])
                combined_shape[kv_cache_seq_dim]=max(right_indices)
                # Set to zero to avoid propagating nans in padded values.
                kv_cache = torch.zeros(combined_shape, dtype=tensors_to_merge[0].dtype, device=device)
                for tensor, start_index, end_index, left_index, right_index in zip(tensors_to_merge, start_indices, end_indices, left_indices, right_indices):
                    kv_cache[[slice(start_index, end_index), *(slice(None) for _ in range(1, kv_cache_seq_dim)), slice(left_index,right_index)]].copy_(tensor)
                if kv_format is None:
                    past_key_values.append(kv_cache)
                elif j==0:
                    past_key_values.append([kv_cache])
                else:
                    past_key_values[-1].append(kv_cache)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            kv_cache_seq_dim=kv_cache_seq_dim,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


class VectorizedNextTokenChooser:
    def __init__(
        self,
        batch_size:int,
        watermark:Optional[List[Optional[bool]]]=None,
        temperature:Optional[List[Optional[float]]]=None,
        repetition_penalty:Optional[List[Optional[float]]]=None,
        top_k:Optional[List[Optional[int]]]=None,
        top_p:Optional[List[Optional[float]]]=None,
        typical_p:Optional[List[Optional[float]]]=None,
        do_sample:Optional[List[Optional[bool]]]=None,
        seeds:Optional[List[Optional[int]]]=None,
        device:torch.device="cpu",
    ):
        self.batch_size=batch_size
        self.filter_value = -float("Inf")
        self.device=device

        # TODO: Seeds are ignored
        self.seeds=self._standardize(seeds, 0)
        self.do_sample=self._standardize(do_sample, False)

        self.watermark=self._standardize(watermark, False)
        if any(self.watermark):
            raise NotImplementedError("Watermarking not implemented")

        self.repetition_penalty=self._standardize(repetition_penalty, 1.0)
        if any([x!=1.0 for x in self.repetition_penalty]):
            self.repetition_penalty_t=torch.tensor(self.repetition_penalty, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            self.repetition_penalty_t=None

        self.temperature=self._standardize(temperature, 1.0)
        if any([x!=1.0 for x in self.temperature]):
            self.do_sample=[sample or x!=1.0 for x, sample in zip(self.temperature, self.do_sample)]
            self.temperature_t=torch.tensor(self.temperature, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            self.temperature_t=None

        self.top_k=self._standardize(top_k, 0)
        n_top_k=sum([x!=0 for x in top_k])
        if n_top_k>0:
            self.do_sample=[sample or x!=0 for x, sample in zip(self.top_k, self.do_sample)]
            self.max_top_k=max(self.top_k)
            self.top_k_t=torch.tensor([max(x-1,0) for x in self.top_k], dtype=torch.int64, device=self.device).unsqueeze(1)
            if n_top_k<self.batch_size:
                self.top_k_mask=torch.tensor([x==0 for x in self.top_k], dtype=torch.bool, device=self.device)
            else:
                self.top_k_mask=None
        else:
            self.max_top_k=None
            self.top_k_t=None
            self.top_k_mask=None

        self.top_p=self._standardize(top_p, 1.0)
        if any([x<1.0 for x in self.top_p]):
            self.do_sample=[sample or x<1.0 for x, sample in zip(temperature, self.top_p)]
            self.top_p_t=torch.tensor([1.0-x for x in self.top_p], dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            self.top_p_t=None

        self.typical_p=self._standardize(typical_p, 1.0)
        if any([x<1.0 for x in self.typical_p]):
            self.do_sample=[sample or x<1.0 for x, sample in zip(self.typical_p, self.do_sample)]
            self.typical_p_t=torch.tensor(self.typical_p, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            self.typical_p_t=None

        self.num_do_sample=sum(self.do_sample)
        if 0<self.num_do_sample<self.batch_size:
            # Mixed greedy and probabilistic sampling. Compute both and pick the right one.
            self.do_sample_t=torch.tensor(self.do_sample, dtype=torch.bool, device=self.device)
        else:
            self.do_sample_t=None

    def _standardize(self, values, default):
        if isinstance(values, list):
            values=values.copy()
        else:
            values=[values]*self.batch_size
        assert len(values)==self.batch_size
        for i, v in enumerate(values):
            if v is None:
                values[i]=default
        return values

    def __call__(self, input_ids, scores):
        # Only process the last token
        scores=scores[: -1, :]

        if self.repetition_penalty_t is not None:
            score = torch.gather(scores, 1, input_ids)
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * self.repetition_penalty_t, score / self.repetition_penalty_t)
            scores.scatter_(1, input_ids, score)

        if self.temperature_t is not None:
            scores.div_(self.temperature_t)

        if self.top_k_t is not None:
            if scores.size(-1)>self.max_top_k: # Safety check
                max_top_k=scores.size(-1)
                top_k=torch.clamp_max(self.top_k_t,max_top_k) # Run only if needed.
            else:
                max_top_k=self.max_top_k
                top_k=self.top_k_t
            kth_scores=torch.gather(torch.topk(scores, max_top_k)[0], 1, top_k)
            if self.top_k_mask is not None:
                kth_scores.masked_fill_(self.top_k_mask, self.filter_value)
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = scores < kth_scores
            scores = scores.masked_fill(indices_to_remove, self.filter_value)

        if self.top_p_t is not None:
            # TODO: Merge wit top_k
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= self.top_p_t
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, self.filter_value)

        if self.typical_p_t is not None:
            # calculate entropy
            normalized = torch.nn.functional.log_softmax(scores, dim=-1)
            p = torch.exp(normalized)
            ent = -(normalized * p).nansum(-1, keepdim=True)
            # shift and sort
            shifted_scores = torch.abs((-normalized) - ent)
            sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
            sorted_logits = scores.gather(-1, sorted_indices)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative mass above the threshold
            last_ind = (cumulative_probs < self.typical_p_t).sum(dim=1)
            last_ind[last_ind < 0] = 0
            sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, self.filter_value)

        # Compute logprobs
        logprobs = torch.log_softmax(scores, dim=-1)

        if self.num_do_sample:
            probs = torch.nn.functional.softmax(scores, -1)
            next_token_ids = torch.multinomial(probs, num_samples=1)
            if self.do_sample_t is not None:
                next_token_ids=torch.where(self.do_sample_t, next_token_ids,torch.argmax(scores, dim=-1))
        else:
            next_token_ids = torch.argmax(scores, dim=-1)

        return next_token_ids, logprobs

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        device: torch.device,
    ) -> "VectorizedNextTokenChooser":
        # TODO: Seeds are ignored
        return VectorizedNextTokenChooser(
            batch_size=len(pb),
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seeds=[pb_.seed for pb_ in pb],
            device=device,
        )

    def filter(self, keep_indices: List[int]) -> "VectorizedNextTokenChooser":
        return VectorizedNextTokenChooser(
            batch_size=len(keep_indices),
            watermark=[self.watermark[i] for i in keep_indices],
            temperature=[self.temperature[i] for i in keep_indices],
            repetition_penalty=[self.repetition_penalty[i] for i in keep_indices],
            top_k=[self.top_k[i] for i in keep_indices],
            top_p=[self.top_p[i] for i in keep_indices],
            typical_p=[self.typical_p[i] for i in keep_indices],
            do_sample=[self.do_sample[i] for i in keep_indices],
            seeds=[self.seeds[i] for i in keep_indices],
            device=self.device,
        )

    @classmethod
    def concatenate(cls, next_token_choosers: List["VectorizedNextTokenChooser"]) -> "VectorizedNextTokenChooser":
        return cls(
            batch_size=sum(next_token_chooser.batch_size for next_token_chooser in next_token_choosers),
            watermark=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.watermark],
            temperature=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.temperature],
            repetition_penalty=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.repetition_penalty],
            top_k=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.top_k],
            top_p=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.top_p],
            typical_p=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.typical_p],
            do_sample=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.do_sample],
            seeds=[x for next_token_chooser in next_token_choosers for x in next_token_chooser.seeds],
            device=next_token_choosers[0].device,
        )




class VectorizedCausalLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: bool = False,
        decode_buffer: int = 3,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # TODO: Choose dtype (fp16?)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize,
            trust_remote_code=True,
        ).eval()
        tokenizer.pad_token_id = (
            self.model.config.pad_token_id
            if self.model.config.pad_token_id is not None
            else self.model.config.eos_token_id
        )

        super().__init__(
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            decode_buffer=decode_buffer,
        )

    @property
    def batch_type(self) -> Type[VectorizedCausalLMBatch]:
        return VectorizedCausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, cleanup_tokenization_spaces=False
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: VectorizedCausalLMBatch
    ) -> Tuple[List[Generation], Optional[VectorizedCausalLMBatch]]:
        key_length=batch.max_input_length
        query_length=key_length if batch.past_key_values is None else 1
        input_ids=batch.input_ids[:, key_length-query_length: key_length]

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=batch.attention_mask[:, : key_length],
            position_ids=batch.position_ids[:, key_length-query_length: key_length],
            past_key_values=batch.past_key_values,
        )
        # TODO: Post-processing
        next_token_ids, logprobs = batch.next_token_chooser(input_ids, outputs.logits[:, -1, :])

        # Update batch
        # TODO: Why do we need all input ids?
        batch.input_ids[:, key_length].copy_(next_token_ids)
        batch.past_key_values=outputs.past_key_values
        batch.input_lengths=[length+1 for length in batch.input_lengths]
        batch.max_input_length+=1

        # TODO: self.decode_token, offsets?
        next_token_ids=next_token_ids.cpu().tolist()
        next_token_texts=self.tokenizer.batch_decode(next_token_ids)

        # TODO: Why do we need logprobs?
        logprobs=logprobs.cpu().tolist()

        # TODO: Vectorize some of this?

        generations: List[Generation] = []
        next_batch=None

        for i, (next_token_id, next_token_text) in enumerate(zip(next_token_ids, next_token_texts)):
            stopping_criterias=batch.stopping_criterias[i]
            stop, reason = stopping_criterias(
                next_token_id,
                next_token_text,
            )
            if stop:
                # Decode generated tokens
                # TODO: Same as stopping_criteria.current_output?
                output_text = self.decode(
                    batch.input_ids[i, -stopping_criterias.current_tokens :]
                )
                # TODO: Seed
                generated_text = GeneratedText(
                    output_text, stopping_criterias.current_tokens, reason, seed=None
                )
            else:
                # Keep request in the batch
                generated_text = None
                next_batch = batch


            generation = Generation(
                batch.requests[i].id,
                None, # TODO: Prefill tokens
                next_token_id,
                logprobs[i],
                next_token_text,
                next_token_id in self.all_special_ids,
                generated_text,
            )

            generations.append(generation)

        return generations, next_batch

