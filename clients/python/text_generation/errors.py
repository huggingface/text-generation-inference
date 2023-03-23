from typing import Dict


# Text Generation Inference Errors
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


# API Inference Errors
class BadRequestError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ShardNotReadyError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ShardTimeoutError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class NotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class NotSupportedError(Exception):
    def __init__(self, model_id: str):
        message = (
            f"Model `{model_id}` is not available for inference with this client. \n"
            "Use `huggingface_hub.inference_api.InferenceApi` instead."
        )
        super(NotSupportedError, self).__init__(message)


# Unknown error
class UnknownError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def parse_error(status_code: int, payload: Dict[str, str]) -> Exception:
    """
    Parse error given an HTTP status code and a json payload

    Args:
        status_code (`int`):
            HTTP status code
        payload (`Dict[str, str]`):
            Json payload

    Returns:
        Exception: parsed exception

    """
    # Try to parse a Text Generation Inference error
    message = payload["error"]
    if "error_type" in payload:
        error_type = payload["error_type"]
        if error_type == "generation":
            return GenerationError(message)
        if error_type == "incomplete_generation":
            return IncompleteGenerationError(message)
        if error_type == "overloaded":
            return OverloadedError(message)
        if error_type == "validation":
            return ValidationError(message)

    # Try to parse a APIInference error
    if status_code == 400:
        return BadRequestError(message)
    if status_code == 403 or status_code == 424:
        return ShardNotReadyError(message)
    if status_code == 504:
        return ShardTimeoutError(message)
    if status_code == 404:
        return NotFoundError(message)
    if status_code == 429:
        return RateLimitExceededError(message)

    # Fallback to an unknown error
    return UnknownError(message)
