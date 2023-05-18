import math
from typing import Optional, List, Union, Dict

from transformers import (
    LogitsWarper,
    LogitsProcessor,
    LogitsProcessorList,
)
import torch

from text_generation_server.pb import generate_pb2
from text_generation_server.utils.tokens import Greedy, Sampling
from text_generation_server.utils.watermark import WatermarkLogitsProcessor


class HeterogeneousRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        repetition_penalty (`List[float]`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: List[float], device: torch.device):
        self.penalty = torch.tensor(
            penalty, dtype=torch.float32, device=device
        ).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores


class HeterogeneousTemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: List[float], device: torch.device):
        self.temperature = torch.tensor(
            temperature, dtype=torch.float32, device=device
        ).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores.div_(self.temperature)
        return scores


class HeterogeneousTopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: List[float],
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.top_p = torch.tensor(top_p, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores


class HeterogeneousTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: List[int],
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.max_top_k = max(top_k)
        self.top_k = torch.tensor(
            [max(x - 1, min_tokens_to_keep - 1) for x in top_k],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)
        zeros = [x == 0 for x in top_k]
        if any(zeros):
            self.top_k_mask = torch.tensor(zeros, dtype=torch.bool, device=device)
        else:
            self.top_k_mask = None
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if scores.size(-1) > self.max_top_k:  # Safety check
            max_top_k = scores.size(-1)
            top_k = torch.clamp_max(self.top_k, max_top_k)  # Run only if needed.
        else:
            max_top_k = self.max_top_k
            top_k = self.top_k
        kth_scores = torch.gather(torch.topk(scores, max_top_k)[0], 1, top_k)
        if self.top_k_mask is not None:
            kth_scores.masked_fill_(self.top_k_mask, self.filter_value)
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < kth_scores
        scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores


class HeterogeneousTypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        mass (`float`):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        mass: List[float],
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.filter_value = filter_value
        self.mass = torch.tensor(mass, dtype=torch.float32, device=device).unsqueeze(1)
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
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
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
            1, last_ind.view(-1, 1)
        )
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        scores = scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores


class HeterogeneousProcessorWrapper(LogitsProcessor):
    r"""
    A wrapper for logit warpers or processors without heterogeneous parameter support.

    Args:
        processors (`Dict[int, Union[LogitsProcessor, LogitsWarper]]`):
            A mapping of sample indices to logit warpers or processors, to be run sequentially.
    """

    def __init__(
        self,
        processors: Dict[int, Union[LogitsProcessor, LogitsWarper]],
    ):
        self.processors = processors
        self.max_index = max(processors)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for i, processor in self.processors.items():
            scores[i : i + 1] = processor(input_ids[i : i + 1], scores[i : i + 1])
        return scores


class HeterogeneousSampling:
    r"""
    Mixed greedy and probabilistic sampling. Compute both and pick the right one for each sample.
    """

    def __init__(self, do_sample: List[bool], seeds: List[int], device: torch.device):
        self.seeds = seeds
        self.greedy = Greedy()
        # TODO: Most seeds are ignored
        self.sampling = Sampling(seeds[0], device)
        self.do_sample = torch.tensor(do_sample, dtype=torch.bool, device=device)

    def __call__(self, logits):
        return torch.where(self.do_sample, self.sampling(logits), self.greedy(logits))


class HeterogeneousNextTokenChooser:
    def __init__(
        self,
        *,
        batch_size: int,
        device: torch.device,
        watermark: Optional[Union[bool, List[Optional[bool]]]] = None,
        temperature: Optional[Union[float, List[Optional[float]]]] = None,
        repetition_penalty: Optional[Union[float, List[Optional[float]]]] = None,
        top_k: Optional[Union[int, List[Optional[int]]]] = None,
        top_p: Optional[Union[float, List[Optional[float]]]] = None,
        typical_p: Optional[Union[float, List[Optional[float]]]] = None,
        do_sample: Optional[Union[bool, List[Optional[bool]]]] = None,
        seeds: Optional[Union[int, List[Optional[int]]]] = None,
    ):
        # TODO: Most seeds are ignored
        seeds = self._standardize(seeds, batch_size, 0)
        do_sample = self._standardize(do_sample, batch_size, False)

        warpers = LogitsProcessorList()

        watermark = self._standardize(watermark, batch_size, False)
        if any(watermark):
            warpers.append(
                HeterogeneousProcessorWrapper(
                    {
                        i: WatermarkLogitsProcessor(device=device)
                        for i, x in watermark
                        if x
                    }
                )
            )

        repetition_penalty = self._standardize(repetition_penalty, batch_size, 1.0)
        if any([x != 1.0 for x in repetition_penalty]):
            warpers.append(
                HeterogeneousRepetitionPenaltyLogitsProcessor(
                    repetition_penalty, device
                )
            )

        temperature = self._standardize(temperature, batch_size, 1.0)
        if any([x != 1.0 for x in temperature]):
            do_sample = [
                sample or x != 1.0 for x, sample in zip(temperature, do_sample)
            ]
            warpers.append(HeterogeneousTemperatureLogitsWarper(temperature, device))

        top_k = self._standardize(top_k, batch_size, 0)
        n_top_k = sum([x != 0 for x in top_k])
        if n_top_k > 0:
            do_sample = [sample or x != 0 for x, sample in zip(top_k, do_sample)]
            warpers.append(HeterogeneousTopKLogitsWarper(top_k, device))

        top_p = self._standardize(top_p, batch_size, 1.0)
        if any([x < 1.0 for x in top_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(top_p, do_sample)]
            warpers.append(HeterogeneousTopPLogitsWarper(top_p, device))

        typical_p = self._standardize(typical_p, batch_size, 1.0)
        if any([x < 1.0 for x in typical_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(typical_p, do_sample)]
            warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, device))

        self.warpers = warpers

        num_do_sample = sum(do_sample)
        if num_do_sample == 0:
            self.choice = Greedy()
        elif num_do_sample < batch_size:
            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        else:
            # TODO: Most seeds are ignored
            self.choice = Sampling(seeds[0], device)

    @staticmethod
    def _standardize(values, batch_size, default):
        if isinstance(values, list):
            values = values.copy()
        else:
            values = [values] * batch_size
        assert len(values) == batch_size
        for i, v in enumerate(values):
            if v is None:
                values[i] = default
        return values

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, return_logprobs: bool
    ):
        last_token_scores = self.warpers(input_ids, scores[:, -1, :])
        next_token_ids = self.choice(last_token_scores)

        if return_logprobs:
            # Compute logprobs
            if scores.size(1) == 1:
                scores = last_token_scores.unsqueeze(1)
            else:
                # TODO: Post-process all the tokens?
                scores[:, -1, :] = last_token_scores
            logprobs = torch.log_softmax(scores, dim=-1)
        else:
            logprobs = None

        return next_token_ids, logprobs

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        device: torch.device,
    ) -> "HeterogeneousNextTokenChooser":
        # TODO: Seeds are ignored
        return HeterogeneousNextTokenChooser(
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
