import re
import torch

from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    PreTrainedTokenizerBase,
)
from typing import List, Tuple, Optional

from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import FinishReason
from text_generation_server.utils.watermark import WatermarkLogitsProcessor


class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        next_tokens = torch.multinomial(probs, num_samples=1, generator=self.generator)
        return next_tokens


class Greedy:
    def __call__(self, logits):
        return logits.argmax()


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
        warpers = LogitsProcessorList()
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        sampling = do_sample

        if watermark:
            warpers.append(WatermarkLogitsProcessor(device=device))
        if repetition_penalty is not None and repetition_penalty != 1.0:
            warpers.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if temperature is not None and temperature != 1.0:
            temperature = float(temperature)
            warpers.append(TemperatureLogitsWarper(temperature))
            sampling = True
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
            sampling = True
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
            sampling = True
        if typical_p is not None and typical_p < 1.0:
            warpers.append(TypicalLogitsWarper(mass=typical_p))
            sampling = True

        self.warpers = warpers
        self.choice = Sampling(seed, device) if sampling else Greedy()

    def __call__(self, input_ids, scores):
        # Warp logits
        if scores.shape[0] > 1:
            # only warp the last token logits
            scores[-1:, :] = self.warpers(input_ids, scores[-1:, :])
        else:
            scores = self.warpers(input_ids, scores)

        # Compute logprobs
        logprobs = torch.log_softmax(scores, -1)

        # Choose tokens
        next_id = self.choice(scores[-1])

        return next_id.view(1, 1), logprobs

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
