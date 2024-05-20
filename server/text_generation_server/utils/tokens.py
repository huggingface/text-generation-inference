# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import re
from typing import List, Optional, Tuple

import math
import torch
from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import FinishReason, GrammarType
from text_generation_server.utils.logits_process import (
    FrequencyPenaltyLogitsProcessor,
    GrammarLogitProcessor,
    HeterogeneousProcessorWrapper,
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousFrequencyPenaltyLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
    HeterogeneousGrammarLogitProcessor,
    static_warper,
)
from text_generation_server.utils.watermark import WatermarkLogitsProcessor
from transformers import PreTrainedTokenizerBase, RepetitionPenaltyLogitsProcessor
from transformers.utils import to_py_obj


class NextTokenChooser:
    def __init__(
        self,
        watermark=False,
        temperature=1.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        top_k=None,
        top_p=None,
        typical_p=None,
        do_sample=False,
        seed=0,
        device="cpu",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        grammar: str = "",
        grammar_type: GrammarType = GrammarType.GRAMMAR_TYPE_NONE,
        fsm_grammar_state: int = 0,
    ):
        self.watermark_processor = (
            WatermarkLogitsProcessor(device=device) if watermark else None
        )
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            if repetition_penalty and repetition_penalty != 1.0
            else None
        )
        self.frequency_processor = (
            FrequencyPenaltyLogitsProcessor(penalty=frequency_penalty)
            if frequency_penalty and frequency_penalty != 0.0
            else None
        )
        self.grammar_processor = (
            GrammarLogitProcessor(tokenizer, device, grammar, grammar_type)
            if grammar != ""
            else None
        )
        self.tokenizer = tokenizer

        has_warpers = (
            (temperature is not None and temperature != 1.0)
            or (top_k is not None and top_k != 0)
            or (top_p is not None and top_p < 1.0)
            or (typical_p is not None and typical_p < 1.0)
        )
        if has_warpers:
            self.static_warper = static_warper(temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p)
        else:
            self.static_warper = None

        sampling = do_sample or has_warpers

        self.choice = Sampling(seed, device) if sampling else Greedy()
        self.fsm_grammar_state = fsm_grammar_state
        self.grammar = grammar

    def __call__(self, input_ids, scores):
        if self.watermark_processor is not None:
            scores = self.watermark_processor(input_ids, scores)
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)
        if self.frequency_processor is not None:
            scores = self.frequency_processor(input_ids, scores)
        if self.grammar_processor is not None:
            scores = self.grammar_processor(scores, self.fsm_grammar_state)

        if self.static_warper is None:
            next_logprob = torch.log_softmax(scores, -1)
        else:
            scores, next_logprob = self.static_warper(scores)

        next_id = self.choice(scores[-1]).view(1, 1)

        return next_id, next_logprob

    def advance_grammar(self, next_id: int):
        if self.grammar_processor is not None:
            self.fsm_grammar_state = self.grammar_processor.advance(
                next_id, self.fsm_grammar_state
            )
        return self

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.NextTokenChooserParameters,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "NextTokenChooser":
        return NextTokenChooser(
            watermark=pb.watermark,
            temperature=pb.temperature,
            repetition_penalty=pb.repetition_penalty,
            frequency_penalty=pb.frequency_penalty,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=device,
            tokenizer=tokenizer,
            grammar=pb.grammar,
            grammar_type=pb.grammar_type,
        )


class StopSequenceCriteria:
    def __init__(self, stop_sequence: str):
        stop_sequence = re.escape(stop_sequence)
        self.regex = re.compile(f"{stop_sequence}$")

    def __call__(self, output: str) -> bool:
        if self.regex.findall(output):
            return True
        return False


class StoppingCriteria:
    def __init__(
        self,
        eos_token_id: int,
        stop_sequence_criterias: List[StopSequenceCriteria],
        max_new_tokens: int = 20,
        ignore_eos_token: bool = False,
    ):
        self.eos_token_id = eos_token_id
        self.stop_sequence_criterias = stop_sequence_criterias
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0
        self.current_output = ""
        self.ignore_eos_token = ignore_eos_token

    def __call__(self, last_token: int, last_output: str) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, FinishReason.FINISH_REASON_LENGTH

        if not self.ignore_eos_token and last_token == self.eos_token_id:
            return True, FinishReason.FINISH_REASON_EOS_TOKEN

        if self.stop_sequence_criterias:
            self.current_output += last_output
            # There is no need to keep an output that is too long
            if len(self.current_output) > 300:
                # Slice to -200 to avoid doing it all the time
                self.current_output = self.current_output[-200:]
            for stop_sequence_criteria in self.stop_sequence_criterias:
                if stop_sequence_criteria(self.current_output):
                    return True, FinishReason.FINISH_REASON_STOP_SEQUENCE

        return False, None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.StoppingCriteriaParameters,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "StoppingCriteria":
        stop_sequence_criterias = [StopSequenceCriteria(sequence) for sequence in pb.stop_sequences]
        return StoppingCriteria(
            tokenizer.eos_token_id,
            stop_sequence_criterias,
            pb.max_new_tokens,
            pb.ignore_eos_token,
        )


def create_n_gram_speculation(
    input_ids: torch.Tensor,
    next_ids: torch.Tensor,
    accepted_ids: torch.Tensor,
    speculate: int,
    verbose: bool,
):
    # Very trivial approach, find first match in the string.
    # This is much less refined than actual n-gram but seems to work
    # relatively OK in grounded mode and is by far much faster with
    # much less worst case complexity as everything happens on device.
    B = accepted_ids.shape[0]
    device = input_ids.device
    seeds = next_ids[accepted_ids.cumsum(dim=-1) - 1]
    indices = (input_ids == seeds.unsqueeze(-1)).max(dim=1).indices + 1
    all_indices = indices.unsqueeze(-1).expand(B, speculate) + torch.arange(
        speculate, device=device
    )
    all_indices = torch.clamp(all_indices, max=input_ids.shape[1] - 1)

    speculative_ids = input_ids.gather(dim=-1, index=all_indices)
    return speculative_ids


class HeterogeneousNextTokenChooser:
    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        watermark: List[bool],
        temperature: List[float],
        repetition_penalty: List[float],
        frequency_penalty: List[float],
        top_k: List[int],
        top_p: List[float],
        typical_p: List[float],
        do_sample: List[bool],
        seeds: List[int],
        tokenizer: PreTrainedTokenizerBase,
        grammars: List[str],
        grammar_types: List[int],
        fsm_grammar_states: List[int],
        quantization_enabled: bool,
    ):
        warpers = []

        # TODO: enable watermark with FP8 quantization
        self.watermark_processor = (
            HeterogeneousProcessorWrapper(
                {
                    i: WatermarkLogitsProcessor(device=device)
                    for i, do_watermark in enumerate(watermark)
                    if do_watermark
                }
            )
            if any(watermark) and not quantization_enabled
            else None
        )

        self.repetition_processor = (
            HeterogeneousRepetitionPenaltyLogitsProcessor(
                repetition_penalty, dtype, device
            )
            if any([x != 1.0 for x in repetition_penalty])
            else None
        )

        self.frequency_processor = (
            HeterogeneousFrequencyPenaltyLogitsProcessor(
                frequency_penalty, dtype, device
            )
            if any([x != 0.0 for x in frequency_penalty])
            else None
        )

        self.grammar_processor = (
            HeterogeneousGrammarLogitProcessor(
                tokenizer, device, grammars, grammar_types
            )
            if any([grammar != "" for grammar in grammars])
            else None
        )

        if any([x != 1.0 for x in temperature]):
            do_sample = [
                sample or x != 1.0 for x, sample in zip(temperature, do_sample)
            ]
            warpers.append(
                HeterogeneousTemperatureLogitsWarper(temperature, dtype, device)
            )

        if any([x != 0 for x in top_k]):
            do_sample = [sample or x != 0 for x, sample in zip(top_k, do_sample)]
            warpers.append(HeterogeneousTopKLogitsWarper(top_k, device))

        if any([x < 1.0 for x in top_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(top_p, do_sample)]
            warpers.append(HeterogeneousTopPLogitsWarper(top_p, dtype, device))

        if any([x < 1.0 for x in typical_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(typical_p, do_sample)]
            warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, dtype, device))

        self.warpers = warpers

        if any(do_sample):
            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        else:
            self.choice = Greedy()

        self.seeds = seeds
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer
        self.fsm_grammar_states = fsm_grammar_states
        self.grammars = grammars
        self.grammar_types = grammar_types

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        speculate: int,
        speculated_ids: Optional[torch.Tensor] = None,
        speculative_scores: Optional[torch.Tensor] = None,
        verbose=False,
    ):
        if speculated_ids is not None:
            B = scores.shape[0] // (speculated_ids.shape[1] + 1)
            S = speculated_ids.shape[1] + 1
            scores = scores.view(B, S, -1)
        else:
            B = scores.shape[0]
            S = 1
            scores = scores.view(B, S, -1)

        next_ids = torch.zeros((B, S), device=scores.device, dtype=torch.long)

        for j in range(S):
            _scores = scores[:, j]
            if self.watermark_processor is not None:
                _scores = self.watermark_processor(input_ids, _scores)
            if self.repetition_processor is not None:
                _scores = self.repetition_processor(input_ids, _scores)
            if self.frequency_processor is not None:
                _scores = self.frequency_processor(input_ids, _scores)
            if self.grammar_processor is not None:
                _scores = self.grammar_processor(_scores, self.fsm_grammar_states)
            for warper in self.warpers:
                _scores = warper(input_ids, _scores)
            _next_ids = self.choice(_scores)
            scores[:, j] = _scores
            next_ids[:, j] = _next_ids
        next_ids = next_ids.view(B * S)
        allscores = scores.view(B * S, -1)
        alllogprobs = torch.log_softmax(allscores, -1)

        if speculated_ids is not None:
            accepted_ids = []
            B = next_ids.shape[0] // (speculated_ids.shape[1] + 1)
            S = speculated_ids.shape[1] + 1
            indices = []
            for i in range(B):
                _next_ids = next_ids[i * S : (i + 1) * S]
                _speculated_ids = speculated_ids[i]
                validate_speculative = _next_ids[:-1] == _speculated_ids
                index = i * S
                accepted = 1
                # First is always valid
                indices.append(index)
                for valid in validate_speculative.tolist():
                    if valid:
                        index += 1
                        accepted += 1
                        indices.append(index)
                    else:
                        break
                accepted_ids.append(accepted)

            accepted_ids = torch.tensor(
                accepted_ids, device=input_ids.device, dtype=input_ids.dtype
            )
            next_ids = next_ids[indices]
            logprobs = alllogprobs[indices]
            indices = torch.arange(B, device=input_ids.device) * S
            if speculative_scores is not None:
                speculative_scores = speculative_scores[indices + accepted_ids - 1]
        else:
            accepted_ids = torch.ones_like(next_ids)
            logprobs = alllogprobs

        next_logprobs = torch.gather(logprobs, 1, next_ids.view(-1, 1)).view(-1)

        if speculate > 0:
            if speculative_scores is not None:
                # Medusa provided some scores
                speculative_ids = Greedy()(speculative_scores)
            else:
                # n-gram
                speculative_ids = create_n_gram_speculation(
                    input_ids, next_ids, accepted_ids, speculate, verbose
                )
        else:
            speculative_ids = None

        return next_ids, next_logprobs, alllogprobs, accepted_ids, speculative_ids

    def advance_grammar(self, next_ids: List[int]):
        if self.grammar_processor is not None:
            other_new_states = self.grammar_processor.advance_batch(
                next_ids, self.fsm_grammar_states
            )
            self.fsm_grammar_states = other_new_states
        return self

    def advance_grammar_single(self, grammar_state_index: int, next_id: int):
        if self.grammar_processor is not None:
            self.fsm_grammar_states[grammar_state_index] = (
                self.grammar_processor.advance_at_index(
                    next_id,
                    self.fsm_grammar_states[grammar_state_index],
                    grammar_state_index,
                )
            )
        return self

    def advance_grammar_single_with_past_state(
        self, grammar_state_index: int, next_id: torch.Tensor, past_state: int
    ):
        if self.grammar_processor is not None:
            next_id = next_id.item()
            self.fsm_grammar_states[grammar_state_index] = (
                self.grammar_processor.advance_at_index(
                    next_id, past_state, grammar_state_index,
                )
            )
        return self

    def filter(self, indices):
        if self.watermark_processor is not None:
            self.watermark_processor = self.watermark_processor.filter(indices)

        if self.repetition_processor is not None:
            self.repetition_processor = self.repetition_processor.filter(indices)

        if self.frequency_processor is not None:
            self.frequency_processor = self.frequency_processor.filter(indices)

        if self.grammar_processor is not None:
            self.grammar_processor = self.grammar_processor.filter(indices)

        filtered_warpers = []
        for warper in self.warpers:
            filtered_warper = warper.filter(indices)
            if filtered_warper is not None:
                filtered_warpers.append(filtered_warper)
        self.warpers = filtered_warpers

        self.seeds = [self.seeds[i] for i in indices]
        self.do_sample = [self.do_sample[i] for i in indices]

        new_grammars = []
        new_fsm_grammar_states = []
        new_grammar_types = []
        for i in indices:
            new_grammars.append(self.grammars[i])
            new_fsm_grammar_states.append(self.fsm_grammar_states[i])
            new_grammar_types.append(self.grammar_types[i])

        self.grammars = new_grammars
        self.fsm_grammar_states = new_fsm_grammar_states
        self.grammar_types = new_grammar_types

        if any(self.do_sample):
            self.choice.filter(indices)
        else:
            self.choice = Greedy()

        return self

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        dtype: torch.dtype,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
        fsm_grammar_states: Optional[List[int]] = None,
        quantization_enabled: bool = False,
    ) -> "HeterogeneousNextTokenChooser":
        return HeterogeneousNextTokenChooser(
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            frequency_penalty=[pb_.frequency_penalty for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seeds=[pb_.seed for pb_ in pb],
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            grammars=[pb_.grammar for pb_ in pb],
            grammar_types=[pb_.grammar_type for pb_ in pb],
            fsm_grammar_states=(
                fsm_grammar_states if fsm_grammar_states else [0] * len(pb)
            ),
            quantization_enabled=quantization_enabled,
        )


class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator("cpu")
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        # Avoid GPU<->CPU sync done by torch multinomial
        # See: https://github.com/pytorch/pytorch/blob/925a3788ec5c06db62ca732a0e9425a26a00916f/aten/src/ATen/native/Distributions.cpp#L631-L637
        q = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return probs.div_(q).argmax()


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class HeterogeneousSampling:
    r"""
    Mixed greedy and probabilistic sampling. Compute both and pick the right one for each sample.
    """

    def __init__(self, do_sample: List[bool], seeds: List[int], device: torch.device):
        self.seeds = seeds

        self.greedy_indices = []
        self.sampling_mapping = {}
        for i, (sample, seed) in enumerate(zip(do_sample, seeds)):
            if sample:
                self.sampling_mapping[i] = Sampling(seed, device)
            else:
                self.greedy_indices.append(i)

        self.greedy = Greedy()

    def __call__(self, logits):
        out = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
        if self.greedy_indices:
            # Computing for all indices is faster than slicing
            torch.argmax(logits, -1, out=out)

        for i, sampling in self.sampling_mapping.items():
            out[i] = sampling(logits[i])
        return out

    def filter(self, indices):
        new_greedy_indices = []
        new_sampling_mapping = {}
        for i, idx in enumerate(indices):
            if idx in self.sampling_mapping:
                new_sampling_mapping[i] = self.sampling_mapping[idx]
            else:
                new_greedy_indices.append(i)

        self.greedy_indices = new_greedy_indices
        self.sampling_mapping = new_sampling_mapping
        return self


def batch_top_tokens(
    top_n_tokens: List[int],
    top_n_tokens_tensor: torch.Tensor,
    logprobs: torch.Tensor,
    accepted_ids: torch.Tensor,
) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
    """Find the top n most likely tokens for a batch of generations.

    When multiple tokens have equal probabilities and they don't all fit, the
    remaining tokens are also returned.
    """
    max_top_n = max(top_n_tokens)
    # Early exit when top_n_tokens is not used
    if max_top_n == 0:
        return [[[]]] * len(top_n_tokens), [[[]]] * len(top_n_tokens)

    batch_size = accepted_ids.shape[0]
    speculate_size = logprobs.shape[0] // batch_size
    top_n_tokens_tensor = top_n_tokens_tensor.repeat_interleave(speculate_size)
    # Ensure top_n doesn't exceed vocab size
    top_n_tokens = [
        min(tok, logprobs.size(-1))
        for tok in top_n_tokens
        for _ in range(speculate_size)
    ]

    # Parallel kthvalue adapted from https://discuss.pytorch.org/t/how-to-efficiently-get-the-k-th-largest-values-in-parallel/160529/2
    # Sorted topk is faster than torch.sort() since we only need a small subset
    sorted_top_k = torch.topk(logprobs, k=max_top_n, dim=-1, sorted=True).values

    nth_highest = torch.gather(
        sorted_top_k, 1, (top_n_tokens_tensor - 1).clip(min=0).unsqueeze(1)
    )
    nth_highest[nth_highest == -float("inf")] = torch.finfo(logprobs.dtype).min

    # Find the new "fuzzy" top n values
    top_n_indices = (logprobs >= nth_highest).nonzero()
    _, top_n_ishes = torch.unique_consecutive(top_n_indices[:, 0], return_counts=True)

    # Take a new topk for these new max n values
    top_k = torch.topk(logprobs, k=top_n_ishes.max(), dim=1, sorted=True)

    top_n_ishes = top_n_ishes.tolist()
    top_indices = top_k.indices.tolist()
    top_values = top_k.values.tolist()

    batch_top_token_ids = []
    batch_top_token_logprobs = []
    accepted_ids_list = accepted_ids.tolist()
    for i, n_accepted_ids in enumerate(accepted_ids_list):
        start = speculate_size * i
        stop = speculate_size * (i + 1)
        _top_indices = top_indices[start:stop]
        _top_values = top_values[start:stop]
        _top_n_ishes = top_n_ishes[start:stop]
        _top_n_tokens = top_n_tokens[start:stop]

        _top_indices = _top_indices[:n_accepted_ids]
        _top_values = _top_values[:n_accepted_ids]
        _top_n_ishes = _top_n_ishes[:n_accepted_ids]
        _top_n_tokens = _top_n_tokens[:n_accepted_ids]

        row_top_token_ids = []
        row_top_token_logprobs = []

        for idxs, vals, n, req_n in zip(
            _top_indices, _top_values, _top_n_ishes, _top_n_tokens
        ):
            indices = idxs[:n] if req_n > 0 else []
            values = vals[:n] if req_n > 0 else []

            row_top_token_ids.append(indices)
            row_top_token_logprobs.append(values)

        batch_top_token_ids.append(row_top_token_ids)
        batch_top_token_logprobs.append(row_top_token_logprobs)

    return batch_top_token_ids, batch_top_token_logprobs


def make_tokenizer_optional(tokenizer):
    class _(type(tokenizer)):
        def __call__(
            self,
            text,
            return_tensors,
            padding,
            return_token_type_ids,
            truncation,
            max_length
        ):
            assert return_tensors == "pt", "inccorrect input arguments when calling TransparentTokenizer"
            assert padding == "max_length" or padding == "longest", "inccorrect input arguments when calling TransparentTokenizer"
            assert return_token_type_ids == False, "inccorrect input arguments when calling TransparentTokenizer"
            assert truncation == True, "inccorrect input arguments when calling TransparentTokenizer"

            def str_token_to_int(i):
                if i == '?':
                    return tokenizer.pad_token_id
                else:
                    return int(i)
            all_tokens = [[str_token_to_int(i.strip()) for i in inner_text.split(',')]
                          for inner_text in text]
            if padding == "longest":
                max_length = max(len(tokens) for tokens in all_tokens)
            return {"input_ids": torch.tensor([[tokenizer.pad_token_id] * (max_length - len(tokens)) + tokens for tokens in all_tokens]),
                    "attention_mask": torch.tensor([[0] * (max_length - len(tokens)) + [1] * len(tokens) for tokens in all_tokens])}

        def decode(
            self,
            token_ids,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> str:
            return ','.join(str(i) for i in to_py_obj(token_ids))

    import os
    if os.getenv("SKIP_TOKENIZER_IN_TGI", "false").lower() == "true":
        tokenizer.__class__ = _
        tokenizer.is_transparent = True


def is_tokenizer_transparent(tokenizer):
    return hasattr(tokenizer, "is_transparent") and tokenizer.is_transparent is True
