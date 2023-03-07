import pytest

from text_generation import __version__
from huggingface_hub.utils import build_hf_headers


@pytest.fixture
def bloom_model():
    return "bigscience/bloom"


@pytest.fixture
def fake_model():
    return "fake/model"


@pytest.fixture
def unsupported_model():
    return "gpt2"


@pytest.fixture
def base_url():
    return "https://api-inference.huggingface.co/models"


@pytest.fixture
def bloom_url(base_url, bloom_model):
    return f"{base_url}/{bloom_model}"


@pytest.fixture
def fake_url(base_url, fake_model):
    return f"{base_url}/{fake_model}"


@pytest.fixture
def unsupported_url(base_url, unsupported_model):
    return f"{base_url}/{unsupported_model}"


@pytest.fixture(scope="session")
def hf_headers():
    return build_hf_headers(
        library_name="text-generation-tests", library_version=__version__
    )
