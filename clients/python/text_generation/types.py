from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional, List

from text_generation.errors import ValidationError


class Parameters(BaseModel):
    do_sample: bool = False
    max_new_tokens: int = 20
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    stop: List[str] = []
    seed: Optional[int]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    watermark: bool = False
    details: bool = False

    @validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v is v <= 0:
            raise ValidationError("`repetition_penalty` must be strictly positive")
        return v

    @validator("seed")
    def valid_seed(cls, v):
        if v is not None and v is v < 0:
            raise ValidationError("`seed` must be positive")
        return v

    @validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`temperature` must be strictly positive")
        return v

    @validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_k` must be strictly positive")
        return v

    @validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1.0):
            raise ValidationError("`top_p` must be > 0.0 and <= 1.0")
        return v


class Request(BaseModel):
    inputs: str
    parameters: Parameters
    stream: bool = False


class PrefillToken(BaseModel):
    id: int
    text: str
    logprob: Optional[float]


class Token(BaseModel):
    id: int
    text: str
    logprob: float
    special: bool


class FinishReason(Enum):
    Length = "length"
    EndOfSequenceToken = "eos_token"
    StopSequence = "stop_sequence"


class Details(BaseModel):
    finish_reason: FinishReason
    generated_tokens: int
    seed: Optional[int]
    prefill: List[PrefillToken]
    tokens: List[Token]


class StreamDetails(BaseModel):
    finish_reason: FinishReason
    generated_tokens: int
    seed: Optional[int]


class Response(BaseModel):
    generated_text: str
    details: Details


class StreamResponse(BaseModel):
    token: Token
    generated_text: Optional[str]
    details: Optional[StreamDetails]
