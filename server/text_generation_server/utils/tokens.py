import re
import torch

from transformers import (
    RepetitionPenaltyLogitsProcessor,
    PreTrainedTokenizerBase, LogitsProcessorList,
)
from typing import List, Tuple, Optional

from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import FinishReason
from text_generation_server.utils.watermark import WatermarkLogitsProcessor
from text_generation_server.utils import Sampling, Greedy
from text_generation_server.utils.logits_process import static_warper, HeterogeneousRepetitionPenaltyLogitsProcessor, \
    HeterogeneousTemperatureLogitsWarper, HeterogeneousTopKLogitsWarper, HeterogeneousTopPLogitsWarper, \
    HeterogeneousTypicalLogitsWarper, HeterogeneousSampling


class NextTokenChooser:
    def __init__(
            self,
            watermark=False,
            temperature=1.0,
            repetition_penalty=1.0,
            top_k=None,
            top_p=None,
            typical_p=None,
            do_sample=False,
            seed=0,
            device="cpu",
    ):
        self.watermark_processor = (
            WatermarkLogitsProcessor(device=device) if watermark else None
        )
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            if repetition_penalty
            else None
        )

        has_warpers = (
                (temperature is not None and temperature != 1.0)
                or (top_k is not None and top_k != 0)
                or (top_p is not None and top_p < 1.0)
                or (typical_p is not None and typical_p < 1.0)
        )
        if has_warpers:
            self.static_warper = static_warper(
                temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p
            )
        else:
            self.static_warper = None

        sampling = do_sample or has_warpers
        self.choice = Sampling(seed, device) if sampling else Greedy()

    def __call__(self, input_ids, scores):
        if self.watermark_processor:
            scores = self.watermark_processor(input_ids, scores)
        if self.repetition_processor:
            scores = self.repetition_processor(input_ids, scores)

        if self.static_warper is None:
            next_logprob = torch.log_softmax(scores, -1)
        else:
            scores, next_logprob = self.static_warper(scores)

        next_id = self.choice(scores[-1]).view(1, 1)

        return next_id, next_logprob

    @classmethod
    def from_pb(
            cls,
            pb: generate_pb2.NextTokenChooserParameters,
            device: torch.device,
    ) -> "NextTokenChooser":
        return NextTokenChooser(
            watermark=pb.watermark,
            temperature=pb.temperature,
            repetition_penalty=pb.repetition_penalty,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=device,
        )


class StopSequenceCriteria:
    def __init__(self, stop_sequence: str):
        stop_sequence = re.escape(stop_sequence)
        self.regex = re.compile(f".*{stop_sequence}$")

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

        self.current_output += last_output
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
        stop_sequence_criterias = [
            StopSequenceCriteria(sequence) for sequence in pb.stop_sequences
        ]
        return StoppingCriteria(
            tokenizer.eos_token_id,
            stop_sequence_criterias,
            pb.max_new_tokens,
            pb.ignore_eos_token,
        )


class HeterogeneousNextTokenChooser:
    def __init__(
            self,
            dtype: torch.dtype,
            device: torch.device,
            watermark: List[bool],
            temperature: List[float],
            repetition_penalty: List[float],
            top_k: List[int],
            top_p: List[float],
            typical_p: List[float],
            do_sample: List[bool],
            seeds: List[int],
    ):
        warpers = LogitsProcessorList()

        if any(watermark):
            raise NotImplementedError("Watermarking not implemented")

        if any([x != 1.0 for x in repetition_penalty]):
            warpers.append(
                HeterogeneousRepetitionPenaltyLogitsProcessor(
                    repetition_penalty, dtype, device
                )
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
            warpers.append(HeterogeneousTopKLogitsWarper(top_k, dtype, device))

        if any([x < 1.0 for x in top_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(top_p, do_sample)]
            warpers.append(HeterogeneousTopPLogitsWarper(top_p, dtype, device))

        if any([x < 1.0 for x in typical_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(typical_p, do_sample)]
            warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, dtype, device))

        self.warpers = warpers

        num_do_sample = sum(do_sample)
        if num_do_sample == 0:
            self.choice = Greedy()
        else:
            self.choice = HeterogeneousSampling(do_sample, seeds, device)

        self.seeds = seeds
        self.do_sample = do_sample

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        last_token_scores = self.warpers(input_ids, scores)
        next_ids = self.choice(last_token_scores)
        next_logprobs = torch.gather(
            torch.log_softmax(last_token_scores, -1), 1, next_ids.view(-1, 1)
        ).view(-1)

        return next_ids, next_logprobs

    def filter(self, indices):
        for warper in self.warpers:
            warper.filter(indices)
        if isinstance(self.choice, HeterogeneousSampling):
            self.choice.filter(indices)
        self.seeds = [self.seeds[i] for i in indices]
        self.do_sample = [self.do_sample[i] for i in indices]
        return self

    @classmethod
    def from_pb(
            cls,
            pb: List[generate_pb2.NextTokenChooserParameters],
            dtype: torch.dtype,
            device: torch.device,
    ) -> "HeterogeneousNextTokenChooser":
        return HeterogeneousNextTokenChooser(
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seeds=[pb_.seed for pb_ in pb],
            device=device,
            dtype=dtype,
        )
