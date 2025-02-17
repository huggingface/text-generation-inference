import pytest

from text_generation import (
    InferenceAPIClient,
    InferenceAPIAsyncClient,
    Client,
    AsyncClient,
)
from text_generation.errors import NotSupportedError, NotFoundError
from text_generation.inference_api import check_model_support, deployed_models


def test_check_model_support(llama_70b, unsupported_model, fake_model):
    assert check_model_support(llama_70b)
    assert not check_model_support(unsupported_model)

    with pytest.raises(NotFoundError):
        check_model_support(fake_model)


def test_deployed_models():
    deployed_models()


def test_client(llama_70b):
    client = InferenceAPIClient(llama_70b)
    assert isinstance(client, Client)


def test_client_unsupported_model(unsupported_model):
    with pytest.raises(NotSupportedError):
        InferenceAPIClient(unsupported_model)


def test_async_client(llama_70b):
    client = InferenceAPIAsyncClient(llama_70b)
    assert isinstance(client, AsyncClient)


def test_async_client_unsupported_model(unsupported_model):
    with pytest.raises(NotSupportedError):
        InferenceAPIAsyncClient(unsupported_model)
