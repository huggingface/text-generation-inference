import copy
import logging
import time
from abc import ABC
from enum import Enum
from typing import List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.generation import GenerationConfig

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.generation import TokenSelector

from .model import get_export_kwargs_from_env
from .pb.generate_pb2 import (
    Batch,
    CachedBatch,
    FinishReason,
    GeneratedText,
    Generation,
    InfoResponse,
    Request,
    Tokens,
)


# Disable optimum-neuron warnings as it seems to block the server after a while
optimum_logger = logging.getLogger("optimum.neuron")
optimum_logger.setLevel("CRITICAL")


class Generator(ABC):
    """An abstract class to represent the workhorse behind TextGenerationService.

    Ideally, it should not rely on protobuf constructs, but in a first step it does.
    Implementations would typically need a model and a tokenizer to implement the Generator methods.
    """

    @property
    def info(self) -> InfoResponse:
        """This should simply return the expected InfoResponse"""
        raise NotImplementedError

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        raise NotImplementedError

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill is called whenever new requests need to be added.

        When this method returns successfully, a decode method will follow
        with both the current and newly prefilled batch(es).

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        raise NotImplementedError

    def decode(self, batches: List[Batch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode after a prefill or another decode."""
        raise NotImplementedError

    def filter(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch"""
        raise NotImplementedError

    def clear(self):
        """Remove all requests from the generator"""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_id: str, revision: Optional[str]):
        """Factory method "a la transformers" """
        raise NotImplementedError


class Slot:
    """Represents a slot in a static batch"""

    class State(Enum):
        EMPTY = 0
        PAUSE = 1
        READY = 2

    def __init__(self, id: int, tokenizer: PreTrainedTokenizerBase):
        self._id = id
        self._tokenizer = tokenizer
        self.clear()

    def clear(self):
        """Clear the slot and mark it as available."""
        self._state = Slot.State.EMPTY
        self._batch_id = None
        self._request_id = None
        self._inputs = ""
        self._truncate = 0
        self._generation_config = None
        self._tokens = []
        self._mask = torch.tensor([])
        self._selector = None
        self._generated_tokens = 0
        self._next_text_token_start = 0
        self._next_text_token_end = 0
        self._generated_text = ""
        self._next_text = ""

    @property
    def id(self) -> int:
        return self._id

    @property
    def state(self) -> "Slot.State":
        return self._state

    @property
    def batch_id(self) -> int:
        return self._batch_id

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def cached_text(self) -> str:
        return self._inputs + self._generated_text

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    def assign(
        self, batch_id: int, request: Request, generation_config: GenerationConfig
    ):
        """Assign a request to a slot.

        Args:
            request (`Request`):
                The request to be assigned. Contains the inputs and tokens selection parameters.
            generation_config (`transformers.GenerationConfig`):
                The base generation config (might be modified by the request generation parameters).
        """
        self._state = Slot.State.READY
        self._batch_id = batch_id
        self._request_id = request.id
        self._inputs = request.inputs
        if request.truncate:
            self._truncate = request.truncate
        self._generation_config = copy.deepcopy(generation_config)
        # Update generation config with request parameters
        self._generation_config.do_sample = request.parameters.do_sample
        if self._generation_config.do_sample:
            if request.parameters.temperature != 0:
                self._generation_config.temperature = request.parameters.temperature
            if request.parameters.top_k != 0:
                self._generation_config.top_k = request.parameters.top_k
            if request.parameters.top_p != 0:
                self._generation_config.top_p = request.parameters.top_p
            if request.parameters.typical_p != 0:
                self._generation_config.typical_p = request.parameters.typical_p
        if request.parameters.repetition_penalty != 0:
            self._generation_config.repetition_penalty = (
                request.parameters.repetition_penalty
            )
        self.seed = request.parameters.seed
        self._generation_config.max_new_tokens = (
            request.stopping_parameters.max_new_tokens
        )
        self._max_new_tokens = self._generation_config.max_new_tokens
        stop_strings = request.stopping_parameters.stop_sequences
        if stop_strings:
            self._generation_config.stop_strings = stop_strings

    def reset(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        selector: TokenSelector,
    ):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`torch.LongTensor`):
                The new input_ids to use to generate the next token.
            attention_mask: (`torch.LongTensor`):
                The new attention_mask to use to generate the next token.
            selector: (`optimum.neuron.generation.TokenSelector`):
                An object implementing the updated token selection logic.
        """
        self._tokens = input_ids.clone()
        self._next_text_token_start = 0
        self._next_text_token_end = torch.numel(self._tokens)
        self._next_text = ""
        self._mask = attention_mask.clone()
        self._selector = selector

    def pause(self, reset_on_pause: bool):
        """Mark the current slot as paused for generation.

        Note that the KV cache for this slot will still be filled.
        """
        if reset_on_pause:
            # Drop the last token as it will be added back when resuming the slot
            self._generated_tokens -= 1
            # Since generated tokens are now part of the prefill, we need to reevaluate
            # max_new_tokens for the next generation
            self._generation_config.max_new_tokens = (
                self._max_new_tokens - self._generated_tokens
            )
        self._state = Slot.State.PAUSE

    def resume(self):
        """Mark the slot as ready for generation."""
        self._state = Slot.State.READY

    def _decode_next_tokens(
        self,
    ) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # We need to include the tokens that produced the last text to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        new_text = self._tokenizer.decode(
            self._tokens[self._next_text_token_start :], skip_special_tokens=False
        )
        if new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            return ""

        # Compare the generated text with the one using only the tokens producing the last one
        last_text = self._tokenizer.decode(
            self._tokens[self._next_text_token_start : self._next_text_token_end],
            skip_special_tokens=False,
        )
        if len(new_text) == len(last_text):
            # Nothing new was actually generated
            return ""
        # Return the decoded text and store its token offsets
        self._next_text_token_start = self._next_text_token_end
        self._next_text_token_end = torch.numel(self._tokens)
        return new_text[len(last_text) :]

    def append(self, next_token: int) -> str:
        """Append a new generated token to this slot

        The new token is added to the list of generated tokens, which impacts
        directly the generated_text and stopped property.

        The new token is however not added immediately to the slot inputs: it will
        be added later on when it has effectively been used to produce the next token.

        Args:
            next_token (`int`):
                The newly generated token.

        Return:
            The corresponding decoded text (if any).
        """
        self._tokens = torch.cat([self._tokens, torch.LongTensor([next_token])])
        self._mask = torch.cat([self._mask, torch.LongTensor([1])])
        self._generated_tokens += 1
        next_text = self._decode_next_tokens()
        # Now that a new token has been generated, we can append the previous one to the generated text
        self._generated_text += self._next_text
        self._next_text = next_text
        return next_text

    def select(
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.LongTensor:
        """Select the next token from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A scalar torch.LongTensor` containing the selected token.
        """
        return self._selector.select(input_ids, logits)[0]

    @property
    def stopped(self) -> bool:
        # Transformers stopping criteria expects a batch of input ids
        input_ids = torch.unsqueeze(self._tokens, dim=0)
        return self._selector.stopping_criteria(input_ids, None)

    @property
    def generated_text(self) -> str:
        return self._generated_text + self._next_text

    @property
    def next_token(self) -> int:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    @property
    def attention_mask(self) -> torch.LongTensor:
        return self._mask

    @property
    def max_token(self) -> int:
        return self._generation_config.max_length

    @property
    def max_new_tokens(self) -> int:
        # The current value of max_new_tokens: might be different of the target max_new_tokens
        # if the slot has been paused and resumed.
        return self._generation_config.max_new_tokens

    @property
    def truncate(self) -> int:
        return self._truncate


class NeuronGenerator(Generator):
    """A Generator for Neuron models."""

    def __init__(
        self,
        model: NeuronModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.rebuild_cache_on_prefill = not self.model.continuous_batching
        # Specify padding and truncation options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer
        self.special_tokens = self.tokenizer.all_special_ids
        self.slots = [Slot(i, tokenizer) for i in range(self.model.batch_size)]
        self.batch_id = 0

    @property
    def info(self) -> InfoResponse:
        """Returns the expected InfoResponse."""
        dtype = getattr(self.model.config, "torch_dtype", "float32")
        return InfoResponse(
            requires_padding=True,
            dtype=str(dtype),
            device_type="xla",
        )

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        # Just check that the warmup request parameters match the model capacity
        batch_size = self.model.batch_size
        if len(batch.requests) > batch_size:
            raise ValueError(
                f"Inconsistent batch_size configuration: Please make sure the batch_size in the compiled model (currently {batch_size}) matches the batch_size passed to TGI.  The compiled model batch_size is usually in the neuron section of the model config.json file. You may also have passed it into optimum-cli during the compilation process.  The batch size for TGI is usually set in the environment as MAX_BATCH_SIZE."
            )
        self.prefill(batch)
        self.clear()
        return self.model.batch_size * self.model.max_length

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill new requests.

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        active_slots = slots[Slot.State.READY]
        empty_slots = slots[Slot.State.EMPTY]
        if len(empty_slots) < len(batch.requests):
            raise ValueError(
                f"Cannot prefill {len(batch.requests)} new request(s) with only {len(empty_slots)} empty slots."
                f" Please align max_batch_size with the static batch size: {self.model.batch_size}."
            )
        # Assign each request to an empty slot
        logger.debug(
            f"Prefilling {len(batch.requests)} new request(s) with {len(empty_slots)} empty slot(s)"
        )
        new_slots = []
        for request in batch.requests:
            slot = empty_slots.pop()
            slot.assign(self.batch_id, request, self.model.generation_config)
            new_slots.append(slot)
            logger.debug(
                f"Request {slot.request_id} assigned to slot {slot.id} with and max_new_tokens {slot.max_new_tokens}"
            )
        if self.rebuild_cache_on_prefill:
            # We will clear pending slots and prefill all slots
            prefill_slots = self.slots
            seq_ids = None
        else:
            # We only need to pass inputs for the new requests
            prefill_slots = new_slots
            seq_ids = torch.tensor([slot.id for slot in prefill_slots])
        # Reconstruct the full inputs (without padding) as seen by the model.
        # This comprises:
        # - the inputs for new requests,
        # - only when rebuilding the cache, the inputs and the generated text that has already
        # been cached (i.e. excluding the last generated token) for unfinished requests.
        inputs = []
        max_length = 0
        for slot in prefill_slots:
            inputs.append(slot.cached_text)
            # Apply truncation, making sure we fit into static dimensions
            if slot.truncate == 0:
                max_length = self.model.max_length
            elif slot.truncate > max_length and slot.truncate < self.model.max_length:
                max_length = slot.truncate
        # Tokenize with padding and truncation
        padded_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = padded_inputs.input_ids
        attention_mask = padded_inputs.attention_mask
        # Pause previously active slots during generation
        next_tokens = []
        for slot in active_slots:
            slot.pause(reset_on_pause=self.rebuild_cache_on_prefill)
            if self.rebuild_cache_on_prefill:
                # The slot will be reset, so we need to store its next token
                next_tokens.append(slot.next_token)
        # Each slot must be reset with the padded inputs and masks
        for i, slot in enumerate(prefill_slots):
            if slot.state != slot.state.EMPTY:
                if slot.truncate > 0 and slot.truncate < input_ids.shape[-1]:
                    # Apply per-request truncation
                    input_ids[i, : -slot.truncate] = self.tokenizer.pad_token_id
                    attention_mask[i, : -slot.truncate] = 0
                slot_input_ids = input_ids[i : i + 1, :]
                # Padded input ids are also required to set logits processors and stopping criterias
                selector = TokenSelector.create(
                    slot_input_ids,
                    slot.generation_config,
                    self.model,
                    self.model.max_length,
                    tokenizer=self.tokenizer,
                    seed=slot.seed,
                )
                slot_input_ids = slot_input_ids.squeeze(dim=0).type(torch.int64)
                slot_attention_mask = attention_mask[i]
                slot.reset(slot_input_ids, slot_attention_mask, selector)
        # Note: when rebuilding cache on prefill, the new tokens on paused slots will be ignored,
        # as they have already been generated and sent back in the last decode.
        model_inputs = self.model.prepare_inputs_for_prefill(
            input_ids, attention_mask, seq_ids
        )
        logits = self.model(**model_inputs)[0]
        generation, next_batch = self._generate_token(
            prefill_slots, self.batch_id, logits, input_ids
        )
        self.batch_id += 1
        # Reactivate previously active slots for the next decode
        for i, slot in enumerate(active_slots):
            slot.resume()
            if self.rebuild_cache_on_prefill:
                # Append back the next token
                slot.append(next_tokens[i])
        logger.debug("Model ready for decoding")
        if next_batch is not None:
            logger.debug(
                f"Next batch is {next_batch.id} with requests: {next_batch.request_ids}"
            )
        return generation, next_batch

    def decode(
        self, batches: List[CachedBatch]
    ) -> Tuple[List[Generation], CachedBatch]:
        """Decode the specified prefilled requests.

        Args:
            batches (`List[CachedBatch]`):
                A list of previous batches containing the prefilled requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        # batches contains a list composed of:
        # - the batch id returned by the last decode,
        # - the batch id(s) returned by the last prefill(s)
        # Batches are always concatenated during prefill, so we can
        # just carry on with decoding. We adopt the id of the first
        # batch in the list as our next batch id.
        next_batch_id = batches[0].id
        request_ids = []
        for batch in batches:
            request_ids += batch.request_ids
        cleared_request_ids = []
        for slot in self.slots:
            if slot.state == slot.State.READY and slot.request_id not in request_ids:
                cleared_request_ids.append(slot.request_id)
                slot.clear()
        if len(cleared_request_ids) > 0:
            logger.info(
                f"Clearing slot for requests {cleared_request_ids} as they are not requested."
            )
        active_slots = [slot for slot in self.slots if slot.state == slot.State.READY]
        if len(active_slots) < len(request_ids):
            raise ValueError(
                "Unable to decode tokens for non-prefilled batches (probably due to a previous failure)"
            )
        if self.model.continuous_batching:
            decode_slots = active_slots
            seq_ids = torch.tensor([slot.id for slot in decode_slots])
        else:
            decode_slots = self.slots
            seq_ids = None
        # Reconstruct input_ids and attention_mask from decode slots
        n_slots = len(decode_slots)
        input_ids = torch.full(
            [n_slots, 1], fill_value=self.tokenizer.eos_token_id, dtype=torch.int64
        )
        max_length = 0
        for slot in decode_slots:
            max_length = max(max_length, slot.attention_mask.size(-1))
        attention_mask = torch.zeros([n_slots, max_length], dtype=torch.int64)
        for i, slot in enumerate(decode_slots):
            if slot.state != Slot.State.EMPTY:
                # input_ids are simply the tokens generated by the last decode or prefill requests (other tokens are cached)
                input_ids[i, 0] = slot.next_token
                attention_mask[i, : slot.attention_mask.size(-1)] = slot.attention_mask
        model_inputs = self.model.prepare_inputs_for_decode(
            input_ids, attention_mask, seq_ids
        )
        logits = self.model(**model_inputs)[0]
        return self._generate_token(decode_slots, next_batch_id, logits, input_ids)

    def _generate_token(
        self,
        slots: List[Slot],
        next_batch_id: int,
        logits: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> Tuple[List[Generation], CachedBatch]:
        generations = []
        active_slots = False
        for i, slot in enumerate(slots):
            if slot.state != Slot.State.READY:
                continue
            request_id = slot.request_id
            next_token_logits = logits[i : i + 1, -1, :]
            slot_input_ids = input_ids[i : i + 1, :]
            next_token = slot.select(slot_input_ids, next_token_logits)
            next_token_text = slot.append(next_token)
            generated_text = None
            finish_reason = None
            if next_token == self.tokenizer.eos_token_id:
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif slot.stopped:
                if slot.generated_tokens == slot.max_new_tokens:
                    finish_reason = FinishReason.FINISH_REASON_LENGTH
                else:
                    finish_reason = FinishReason.FINISH_REASON_STOP_SEQUENCE
            if finish_reason is not None:
                # We must include the generated text for each finished sequence in the response
                generated_text = GeneratedText(
                    text=slot.generated_text,
                    generated_tokens=slot.generated_tokens,
                    finish_reason=finish_reason,
                )
                logger.debug(
                    f"Decode complete for request {request_id} with {slot.generated_tokens} tokens"
                )
                # mark the slot as available
                slot.clear()
            else:
                active_slots = True
            generations.append(
                Generation(
                    request_id=request_id,
                    prefill_tokens=None,
                    tokens=Tokens(
                        ids=[next_token],
                        logprobs=[0],
                        texts=[next_token_text],
                        is_special=[next_token in self.special_tokens],
                    ),
                    generated_text=generated_text,
                )
            )
        batch = None
        if active_slots:
            # Whatever initial batch these requests came from, we always return all pending requests in a single batch
            request_ids = [
                slot.request_id for slot in self.slots if slot.state == Slot.State.READY
            ]
            batch = self._cached_batch(next_batch_id, request_ids)
        else:
            logger.debug("No more pending requests")
        return generations, batch

    def _cached_batch(self, batch_id: int, request_ids: List):
        size = len(request_ids)
        max_tokens = size * self.model.max_length
        return CachedBatch(
            id=batch_id, request_ids=request_ids, size=size, max_tokens=max_tokens
        )

    def filter(self, batch_id: int, keep_request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch

        Args:
            batch_id (`int`):
                The id of a cached batch.
            keep_ids(`List[int]`):
                The list of requests that must be kept.

        Return:
            A `CachedBatch` containing the pending requests.
        """
        keep_slot_ids = [
            slot.id for slot in self.slots if slot.request_id in keep_request_ids
        ]
        self._clear(keep_slot_ids)
        return self._cached_batch(batch_id, keep_request_ids)

    def clear(self, batch_id: Optional[int] = None):
        """Remove a subset or all requests from the generator"""
        keep_ids = []
        if batch_id is not None:
            keep_ids = [slot.id for slot in self.slots if slot.batch_id != batch_id]
        return self._clear(keep_ids)

    def _clear(self, keep_slot_ids: List):
        for slot in self.slots:
            if slot.state != Slot.State.EMPTY and slot.id not in keep_slot_ids:
                logger.debug(f"Removing slot {slot.id} with request {slot.request_id}")
                slot.clear()

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str = None):
        """Instantiate a NeuronGenerator.

        Args:
            model_id (`str`):
                A hub model id or the path to a local model. This path must also contain a Tokenizer.
            revision (`Optional[str]`, defaults to `None`):
                The revision of the model on the HuggingFace hub.

        Returns:
            A NeuronGenerator.
        """
        config = AutoConfig.from_pretrained(model_id)
        neuron_config = getattr(config, "neuron", None)
        start = time.time()
        if neuron_config is None:
            export_kwargs = get_export_kwargs_from_env()
            logger.info(f"Exporting model to neuron with config: {export_kwargs}.")
            model = NeuronModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                low_cpu_mem_usage=True,
                export=True,
                **export_kwargs,
            )
        else:
            logger.info(
                "Loading model on neuron devices (this can take a few minutes)."
            )
            model = NeuronModelForCausalLM.from_pretrained(
                model_id, low_cpu_mem_usage=True, revision=revision
            )
        end = time.time()
        logger.info(f"Model successfully loaded in {end - start:.2f} s.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return cls(model, tokenizer)
