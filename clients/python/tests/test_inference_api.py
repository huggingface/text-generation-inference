import pytest

from text_generation import (
    InferenceAPIClient,
    InferenceAPIAsyncClient,
    Client,
    AsyncClient,
)
from text_generation.errors import NotSupportedError
from text_generation.inference_api import get_supported_models


def test_get_supported_models():
    assert isinstance(get_supported_models(), list)


def test_client(bloom_model):
    client = InferenceAPIClient(bloom_model)
    assert isinstance(client, Client)


def test_client_unsupported_model(unsupported_model):
    with pytest.raises(NotSupportedError):
        InferenceAPIClient(unsupported_model)


def test_async_client(bloom_model):
    client = InferenceAPIAsyncClient(bloom_model)
    assert isinstance(client, AsyncClient)


def test_async_client_unsupported_model(unsupported_model):
    with pytest.raises(NotSupportedError):
        InferenceAPIAsyncClient(unsupported_model)
