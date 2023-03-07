from text_generation.errors import (
    parse_error,
    GenerationError,
    IncompleteGenerationError,
    OverloadedError,
    ValidationError,
    BadRequestError,
    ShardNotReadyError,
    ShardTimeoutError,
    NotFoundError,
    RateLimitExceededError,
    UnknownError,
)


def test_generation_error():
    payload = {"error_type": "generation", "error": "test"}
    assert isinstance(parse_error(400, payload), GenerationError)


def test_incomplete_generation_error():
    payload = {"error_type": "incomplete_generation", "error": "test"}
    assert isinstance(parse_error(400, payload), IncompleteGenerationError)


def test_overloaded_error():
    payload = {"error_type": "overloaded", "error": "test"}
    assert isinstance(parse_error(400, payload), OverloadedError)


def test_validation_error():
    payload = {"error_type": "validation", "error": "test"}
    assert isinstance(parse_error(400, payload), ValidationError)


def test_bad_request_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(400, payload), BadRequestError)


def test_shard_not_ready_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(403, payload), ShardNotReadyError)
    assert isinstance(parse_error(424, payload), ShardNotReadyError)


def test_shard_timeout_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(504, payload), ShardTimeoutError)


def test_not_found_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(404, payload), NotFoundError)


def test_rate_limit_exceeded_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(429, payload), RateLimitExceededError)


def test_unknown_error():
    payload = {"error": "test"}
    assert isinstance(parse_error(500, payload), UnknownError)
