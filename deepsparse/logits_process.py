import numpy as np
from typing import List

def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x) # TODO: make this happen in place?
    return y / y.sum(axis=axis, keepdims=True)

class LogitsWarper:
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class LogitsProcessor:
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
 
class LogitsWarpers:
    def __init__(
        self, 
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        self.warpers = []
        
        if temperature is not None and temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(temperature=temperature))
        if top_k is not None and top_k != 0:
            self.warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p=top_p))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        for warper in self.warpers:
            scores = warper(scores)
        return scores

# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L205
# note: this handles arbitrary batch size
class TemperatureLogitsWarper(LogitsWarper):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)
        
        self.temperature = temperature

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = scores / self.temperature
        return scores

# https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/generation/logits_process.py#L361
# TODO: update logic to handle b > 1
class TopPLogitsWarper(LogitsWarper):
    def __init__(self, top_p: float):
        if not isinstance(top_p, float) or top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        
        self.top_p = top_p
    
    def __call__(self, scores: np.ndarray):
        # assert shape is [1, vocab_size]
        assert len(scores.shape) == 2
        assert scores.shape[0] == 1

        sorted_indices = np.argsort(scores, axis=-1)
        sorted_scores = np.take_along_axis(scores, sorted_indices, axis=-1)
        
        # sort, grabbing all those outside top_p
        cumulative_probs = softmax(sorted_scores, axis=-1).cumsum(axis=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # note: this relies on b=1
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        
        # set removed indices logits to -Inf (never selected)
        scores[:, indices_to_remove] = -float("Inf")

        return scores

# https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/generation/logits_process.py#L433
# note: this handles arbitrary batch size
class TopKLogitsWarper(LogitsWarper):
    def __init__(self, top_k: int):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        
        self.top_k = top_k
    
    def __call__(self, scores: np.ndarray):
        # assert shape is [batch, vocab_size]
        assert len(scores.shape) == 2

        top_k = min(self.top_k, scores.shape[-1])

        # sort, grabbing all except the top k
        indices_to_remove = np.argsort(scores, axis=-1)[:,:-top_k]
        
        # set removed indices logits to -Inf (never selected)
        np.put_along_axis(scores, indices_to_remove, -float("Inf"), axis=-1)
        
        return scores

# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L270  
# TODO: update logic to handle b > 1
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        # assert shape is [1, vocab_size]
        assert len(scores.shape) == 2
        assert scores.shape[0] == 1

        # assert shape is [1, seq_len]
        assert len(input_ids.shape) == 2
        assert input_ids.shape[0] == 1
        
        score = scores[:, input_ids[0]]     # torch.gather
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores[:, input_ids[0]] = score     # torch.scatter

        return scores

