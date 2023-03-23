import torch

from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)
from typing import List, Tuple, Optional

from text_generation_server.pb import generate_pb2
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
