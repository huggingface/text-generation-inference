import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict
from loguru import logger

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling

tracer = trace.get_tracer(__name__)


@dataclass
class VectorizedCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    input_ids: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    offsets: List[Optional[int]]
    token_offsets: List[Optional[int]]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    max_input_length: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

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
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        offsets = []
        token_offsets = []
        requests_idx_mapping = {}

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0
        for i, r in enumerate(pb.requests):
            next_token_chooser=NextTokenChooser.from_pb(r.parameters, device)
            # TODO: Implement
            assert len(next_token_chooser.warpers)==0
            requests_idx_mapping[r.id] = i
            inputs.append(r.inputs)
            offsets.append(None)
            token_offsets.append(None)
            next_token_choosers.append(next_token_chooser)
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
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

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        input_shape=(pb.size, max_input_length + padding_right_offset)

        # Allocate maximum attention_mask
        attention_mask = torch.empty(input_shape, dtype=torch.bool, device=device)
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length].copy_(tokenized_inputs["attention_mask"])
        attention_mask[:, max_input_length:].fill_(1)

        position_ids = attention_mask.cumsum(-1).sub_(1)
        position_ids[:, :max_input_length].relu_()

        input_ids = torch.empty(input_shape, dtype=torch.int64, device=device)
        input_ids[:, :max_input_length].copy_(tokenized_inputs["input_ids"])

        max_tokens = len(inputs) * max_input_length + max_decode_tokens

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
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length.item(),
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, requests: List[generate_pb2.Request]) -> Optional["VectorizedCausalLMBatch"]:
        raise NotImplementedError()

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["VectorizedCausalLMBatch"]) -> "VectorizedCausalLMBatch":
        raise NotImplementedError()

    def __len__(self):
        return len(self.requests)


class VectorizedNextTokenChooser:
    def __init__(
        self,
        batch_size:int,
        watermark=None,
        temperature=None,
        repetition_penalty=None,
        top_k=None,
        top_p=None,
        typical_p=None,
        do_sample=None,
        seed:int=0,
        device="cpu",
    ):
        self.batch_size=batch_size

        do_sample=self._standardize(do_sample, False)

        watermark=self._standardize(watermark, False)
        if any(watermark):
            raise NotImplementedError("Watermarking not implemented")

        repetition_penalty=self._standardize(repetition_penalty, 1.0)
        if any([x!=1.0 for x in repetition_penalty]):
            self.repetition_penalty=torch.tensor(repetition_penalty, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            self.repetition_penalty=None

        temperature=self._standardize(temperature, 1.0)
        if any([x!=1.0 for x in temperature]):
            do_sample=[sample or x!=1.0 for x, sample in zip(temperature, do_sample)]
            self.temperature=torch.tensor(temperature, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            self.temperature=None

        top_k=self._standardize(top_k, 0)
        n_top_k=sum([x!=0 for x in top_k])
        if n_top_k>0:
            do_sample=[sample or x!=0 for x, sample in zip(top_k, do_sample)]
            self.max_top_k=max(top_k)
            self.top_k=torch.tensor([max(x-1,0) for x in top_k], dtype=torch.float32, device=device).unsqueeze(1)
            if n_top_k<self.batch_size:
                self.top_k_mask=torch.tensor([x==0 for x in top_k], dtype=torch.bool, device=device)
            else:
                self.top_k_mask=None
        else:
            self.max_top_k=None
            self.top_k=None
            self.top_k_mask=None


        top_p=self._standardize(top_p, 1.0)
        if any([x<1.0 for x in top_p]):
            raise NotImplementedError("Top P not implemented")

        typical_p=self._standardize(typical_p, 1.0)
        if any([x<1.0 for x in typical_p]):
            raise NotImplementedError("Typical P not implemented")

        self.do_sample = any(do_sample)
        if self.do_sample and not all(do_sample):
            raise NotImplementedError("Mixed greedy and probabilistic sampling not supported")

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

        if self.repetition_penalty is not None:
            score = torch.gather(scores, 1, input_ids)
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
            scores.scatter_(1, input_ids, score)

        if self.temperature is not None:
            scores.div_(self.temperature)

        if self.top_k is not None:
            if scores.size(-1)>self.max_top_k: # Safety check
                max_top_k=scores.size(-1)
                top_k=torch.clamp_max(self.top_k,max_top_k) # Run only if needed.
            else:
                max_top_k=self.max_top_k
                top_k=self.top_k
            kth_scores=torch.gather(torch.topk(scores, max_top_k)[0], 1, top_k)
            if self.top_k_mask is not None:
                kth_scores.masked_fill_(self.top_k_mask, self.filter_value)
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = scores < kth_scores
            scores = scores.masked_fill(indices_to_remove, self.filter_value)

        # Compute logprobs
        logprobs = torch.log_softmax(scores, dim=-1)

        if self.do_sample:
            raise NotImplementedError()
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
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seed=0,
            device=device,
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

        outputs = self.model.forward(
            input_ids=batch.input_ids[:, key_length-query_length: key_length],
            attention_mask=batch.attention_mask[:, : key_length],
            position_ids=batch.position_ids[:, key_length-query_length: key_length],
            past_key_values=batch.past_key_values,
        )
        # TODO: Post-processing
        next_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        # Update batch
        # TODO: Why do we need all input ids?
        batch.input_ids[:, key_length].copy_(next_token_ids)
        batch.past_key_values=outputs.past_key_values
        batch.input_lengths=[length+1 for length in batch.input_lengths]
        batch.max_input_length+=1

        # TODO: self.decode_token, offsets?
        next_token_ids=next_token_ids.cpu().tolist()
        next_token_texts=self.tokenizer.batch_decode(next_token_ids)

        # TODO: Vectorize some of this?

        generations: List[Generation] = []
        next_batch=None

        for i, (next_token_id, next_token_text) in enumerate(zip(next_token_ids, next_token_texts)):
            stopping_criterias=batch.stopping_criterias[i]
            next_token_chooser=batch.next_token_choosers[i]
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
                # Get seed
                if isinstance(next_token_chooser.choice, Sampling):
                    seed = next_token_chooser.choice.seed
                else:
                    seed = None

                generated_text = GeneratedText(
                    output_text, stopping_criterias.current_tokens, reason, seed
                )
            else:
                # Keep request in the batch
                generated_text = None
                next_batch = batch


            generation = Generation(
                batch.requests[i].id,
                None,
                next_token_id,
                0,
                next_token_text,
                next_token_id in self.all_special_ids,
                generated_text,
            )

            generations.append(generation)

        return generations, next_batch
