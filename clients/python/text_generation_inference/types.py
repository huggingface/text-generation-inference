from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional, List, Type


class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class GenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class OverloadedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IncompleteGenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InferenceAPIError(Exception):
    def __init__(self, message: str):
        super(InferenceAPIError, self).__init__(message)


class ErrorType(str, Enum):
    generation = "generation"
    incomplete_generation = "incomplete_generation"
    overloaded = "overloaded"
    validation = "validation"

    def to_exception_type(self) -> Type[Exception]:
        if self == ErrorType.generation:
            return GenerationError
        if self == ErrorType.incomplete_generation:
            return IncompleteGenerationError
        if self == ErrorType.overloaded:
            return OverloadedError
        if self == ErrorType.validation:
            return ValidationError
        raise ValueError("Unknown error")


class ErrorModel(BaseModel):
    error_type: Optional[ErrorType]
    error: str

    def to_exception(self) -> Exception:
        if self.error_type is not None:
            return self.error_type.to_exception_type()(self.error)
        return InferenceAPIError(self.error)


class Parameters(BaseModel):
    do_sample: bool = False
    max_new_tokens: int = 20
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    seed: Optional[int]
    stop: Optional[List[str]]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    watermark: bool = False

    @validator("seed")
    def valid_seed(cls, v):
        if v is not None and v is v < 0:
            raise ValidationError("`seed` must be strictly positive")
        return v

    @validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`temperature` must be strictly positive")
        return v

    @validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`top_k` must be strictly positive")
        return v

    @validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1.0):
            raise ValidationError("`top_p` must be > 0.0 and <= 1.0")
        return v


class Response(BaseModel):
    generated_text: str


class Token(BaseModel):
    id: int
    text: str
    logprob: float
    special: bool


class StreamResponse(BaseModel):
    token: Token
