from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Optional, List, Union, Any

from text_generation.errors import ValidationError


# enum for grammar type
class GrammarType(str, Enum):
    Json = "json"
    Regex = "regex"


# Grammar type and value
class Grammar(BaseModel):
    # Grammar type
    type: GrammarType
    # Grammar value
    value: Union[str, dict]


class ToolCall(BaseModel):
    # Id of the tool call
    id: int
    # Type of the tool call
    type: str
    # Function details of the tool call
    function: dict


class Message(BaseModel):
    # Role of the message sender
    role: str
    # Content of the message
    content: Optional[str] = None
    # Optional name of the message sender
    name: Optional[str] = None
    # Tool calls associated with the chat completion
    tool_calls: Optional[Any] = None


class Tool(BaseModel):
    # Type of the tool
    type: str
    # Function details of the tool
    function: dict


class ChatCompletionComplete(BaseModel):
    # Index of the chat completion
    index: int
    # Message associated with the chat completion
    message: Message
    # Log probabilities for the chat completion
    logprobs: Optional[Any]
    # Reason for completion
    finish_reason: str
    # Usage details of the chat completion
    usage: Optional[Any] = None


class Function(BaseModel):
    name: Optional[str]
    arguments: str


class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: str
    type: str
    function: Function


class ChoiceDelta(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[ChoiceDeltaToolCall]


class Choice(BaseModel):
    index: int
    delta: ChoiceDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Choice]


class ChatComplete(BaseModel):
    # Chat completion details
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatCompletionComplete]
    usage: Any


class ChatRequest(BaseModel):
    # Model identifier
    model: str
    # List of messages in the conversation
    messages: List[Message]
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # The parameter for frequency penalty. 1.0 means no penalty
    # Penalize new tokens based on their existing frequency in the text so far,
    # decreasing the model's likelihood to repeat the same line verbatim.
    frequency_penalty: Optional[float] = None
    # Bias values for token selection
    logit_bias: Optional[List[float]] = None
    # Whether to return log probabilities
    logprobs: Optional[bool] = None
    # Number of most likely tokens to return at each position
    top_logprobs: Optional[int] = None
    # Maximum number of tokens to generate
    max_tokens: Optional[int] = None
    # Number of chat completion choices to generate
    n: Optional[int] = None
    # Penalty for presence of new tokens
    presence_penalty: Optional[float] = None
    # Flag to indicate streaming response
    stream: bool = False
    # Random sampling seed
    seed: Optional[int] = None
    # Sampling temperature
    temperature: Optional[float] = None
    # Top-p value for nucleus sampling
    top_p: Optional[float] = None
    # List of tools to be used
    tools: Optional[List[Tool]] = None
    # Choice of tool to be used
    tool_choice: Optional[str] = None


class Parameters(BaseModel):
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # The parameter for frequency penalty. 1.0 means no penalty
    # Penalize new tokens based on their existing frequency in the text so far,
    # decreasing the model's likelihood to repeat the same line verbatim.
    frequency_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int] = None
    # The value used to module the logits distribution.
    temperature: Optional[float] = None
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int] = None
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float] = None
    # truncate inputs tokens to the given size
    truncate: Optional[int] = None
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float] = None
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int] = None
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False
    # Return the N most likely tokens at each step
    top_n_tokens: Optional[int] = None
    # grammar to use for generation
    grammar: Optional[Grammar] = None

    @field_validator("best_of")
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValidationError("`best_of` must be strictly positive")
            if field_value > 1 and values.data["seed"] is not None:
                raise ValidationError("`seed` must not be set when `best_of` is > 1")
            sampling = (
                values.data["do_sample"]
                | (values.data["temperature"] is not None)
                | (values.data["top_k"] is not None)
                | (values.data["top_p"] is not None)
                | (values.data["typical_p"] is not None)
            )
            if field_value > 1 and not sampling:
                raise ValidationError("you must use sampling when `best_of` is > 1")

        return field_value

    @field_validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`repetition_penalty` must be strictly positive")
        return v

    @field_validator("frequency_penalty")
    def valid_frequency_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`frequency_penalty` must be strictly positive")
        return v

    @field_validator("seed")
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`seed` must be positive")
        return v

    @field_validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`temperature` must be strictly positive")
        return v

    @field_validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_k` must be strictly positive")
        return v

    @field_validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`top_p` must be > 0.0 and < 1.0")
        return v

    @field_validator("truncate")
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`truncate` must be strictly positive")
        return v

    @field_validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`typical_p` must be > 0.0 and < 1.0")
        return v

    @field_validator("top_n_tokens")
    def valid_top_n_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_n_tokens` must be strictly positive")
        return v

    @field_validator("grammar")
    def valid_grammar(cls, v):
        if v is not None:
            if v.type == GrammarType.Regex and not v.value:
                raise ValidationError("`value` cannot be empty for `regex` grammar")
            if v.type == GrammarType.Json and not v.value:
                raise ValidationError("`value` cannot be empty for `json` grammar")
        return v


class Request(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False

    @field_validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v

    @field_validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values.data["parameters"]
        if (
            parameters is not None
            and parameters.best_of is not None
            and parameters.best_of > 1
            and field_value
        ):
            raise ValidationError(
                "`best_of` != 1 is not supported when `stream` == True"
            )
        return field_value


# Decoder input tokens
class InputToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float] = None


# Generated tokens
class Token(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: Optional[float] = None
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason
class FinishReason(str, Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
class BestOfSequence(BaseModel):
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Most likely tokens
    top_tokens: Optional[List[List[Token]]] = None


# `generate` details
class Details(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Most likely tokens
    top_tokens: Optional[List[List[Token]]] = None
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]] = None


# `generate` return value
class Response(BaseModel):
    # Generated text
    generated_text: str
    # Generation details
    details: Details


# `generate_stream` details
class StreamDetails(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None


# `generate_stream` return value
class StreamResponse(BaseModel):
    # Generated token
    token: Token
    # Most likely tokens
    top_tokens: Optional[List[Token]] = None
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str] = None
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails] = None


# Inference API currently deployed model
class DeployedModel(BaseModel):
    model_id: str
    sha: str
