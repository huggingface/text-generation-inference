import math
import torch

from functools import lru_cache
from typing import Optional, List, Dict, Union

from transformers import (
    LogitsWarper,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)


class StaticWarper:
    def __init__(
        self,
        temperature=1.0,
        top_k=None,
        top_p=None,
        typical_p=None,
    ):
        self.warpers = []

        if temperature is not None and temperature != 1.0:
            temperature = float(temperature)
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            self.warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p=top_p))
        if typical_p is not None and typical_p < 1.0:
            self.warpers.append(TypicalLogitsWarper(mass=typical_p))

        self.cuda_graph = None
        self.static_scores = None
        self.static_warped_scores = None
        self.static_next_logprob = None

    def __call__(self, scores):
        if self.cuda_graph is None:
            self.static_scores = scores
            self.cuda_graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.cuda_graph):
                local_scores = self.static_scores
                for warper in self.warpers:
                    local_scores = warper(None, local_scores)

                self.static_warped_scores = local_scores
                # Compute logprobs
                self.static_next_logprob = torch.log_softmax(
                    self.static_warped_scores, -1
                )

        self.static_scores.copy_(scores)
        self.cuda_graph.replay()

        return self.static_warped_scores, self.static_next_logprob


@lru_cache(10)
def static_warper(
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    typical_p: Optional[float],
) -> StaticWarper:
    return StaticWarper(
        temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p
    )


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

    def __init__(self, penalty: List[float], dtype: torch.dtype, device: torch.device):
        self.penalty = torch.tensor(penalty, dtype=dtype, device=device).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores

    def filter(self, indices):
        self.penalty = self.penalty[indices]
        return self


class HeterogeneousTemperatureLogitsWarper:
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(
        self, temperature: List[float], dtype: torch.dtype, device: torch.device
    ):
        self.temperature = torch.tensor(
            temperature, dtype=dtype, device=device
        ).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores.div_(self.temperature)
        return scores

    def filter(self, indices):
        self.temperature = self.temperature[indices]
        return self


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
        dtype: torch.dtype,
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.top_p_opposite = 1 - torch.tensor(
            top_p, dtype=dtype, device=device
        ).unsqueeze(1)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= self.top_p_opposite
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores

    def filter(self, indices):
        self.top_p_opposite = self.top_p_opposite[indices]
        return self


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
        self.top_k = top_k
        self.max_top_k = max(top_k)
        # value - 1 as we will use top_k to index and python uses 0 based numbering
        self.top_k_tensor = torch.tensor(
            [max(x - 1, min_tokens_to_keep - 1) for x in top_k],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)

        # 0 is a special value that disables top_k warping for this member of the batch
        disabled = [x == 0 for x in top_k]

        if any(disabled):
            self.top_k_disabled_mask = torch.tensor(
                disabled, dtype=torch.bool, device=device
            )
        else:
            self.top_k_disabled_mask = None

        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # If max_top_k is superior to the vocab, we need to clamp or the warper will fail
        if scores.size(-1) < self.max_top_k:
            max_top_k = scores.size(-1)
            top_k = torch.clamp_max(self.top_k_tensor, max_top_k)
        else:
            max_top_k = self.max_top_k
            top_k = self.top_k_tensor

        # Get the kth score for each member of the batch
        kth_scores = torch.gather(torch.topk(scores, max_top_k)[0], 1, top_k)

        # Mask member of kth_scores that do not want to use top_k warping
        if self.top_k_disabled_mask is not None:
            kth_scores.masked_fill_(self.top_k_disabled_mask, self.filter_value)

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < kth_scores
        scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores

    def filter(self, indices):
        self.top_k_tensor = self.top_k_tensor[indices]
        self.top_k = [self.top_k[i] for i in indices]
        self.max_top_k = max(self.top_k)
        return self


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
        dtype: torch.dtype,
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.filter_value = filter_value
        self.mass = torch.tensor(mass, dtype=dtype, device=device).unsqueeze(1)
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

    def filter(self, indices):
        self.mass = self.mass[indices]
        return self


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

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for i, processor in self.processors.items():
            scores[i : i + 1] = processor(input_ids[i : i + 1], scores[i : i + 1])
        return scores

    def filter(self, indices):
        new_processors = {}
        for i, idx in enumerate(indices):
            if idx in self.processors:
                new_processors[i] = self.processors[idx]

        self.processors = new_processors
        return self
