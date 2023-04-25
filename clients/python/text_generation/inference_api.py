import os
import requests

from typing import Dict, Optional, List
from huggingface_hub.utils import build_hf_headers

from text_generation import Client, AsyncClient, __version__
from text_generation.types import DeployedModel
from text_generation.errors import NotSupportedError, parse_error

INFERENCE_ENDPOINT = os.environ.get(
    "HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co"
)


def deployed_models(headers: Optional[Dict] = None) -> List[DeployedModel]:
    """
    Get all currently deployed models with text-generation-inference-support

    Returns:
        List[DeployedModel]: list of all currently deployed models
    """
    resp = requests.get(
        f"https://api-inference.huggingface.co/framework/text-generation-inference",
        headers=headers,
        timeout=5,
    )

    payload = resp.json()
    if resp.status_code != 200:
        raise parse_error(resp.status_code, payload)

    models = [DeployedModel(**raw_deployed_model) for raw_deployed_model in payload]
    return models


def check_model_support(repo_id: str, headers: Optional[Dict] = None) -> bool:
    """
    Check if a given model is supported by text-generation-inference

    Returns:
        bool: whether the model is supported by this client
    """
    resp = requests.get(
        f"https://api-inference.huggingface.co/status/{repo_id}",
        headers=headers,
        timeout=5,
    )

    payload = resp.json()
    if resp.status_code != 200:
        raise parse_error(resp.status_code, payload)

    framework = payload["framework"]
    supported = framework == "text-generation-inference"
    return supported


class InferenceAPIClient(Client):
    """Client to make calls to the HuggingFace Inference API.

     Only supports a subset of the available text-generation or text2text-generation models that are served using
     text-generation-inference

     Example:

     ```python
     >>> from text_generation import InferenceAPIClient

     >>> client = InferenceAPIClient("bigscience/bloomz")
     >>> client.generate("Why is the sky blue?").generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(self, repo_id: str, token: Optional[str] = None, timeout: int = 10):
        """
        Init headers and API information

        Args:
            repo_id (`str`):
                Id of repository (e.g. `bigscience/bloom`).
            token (`str`, `optional`):
                The API token to use as HTTP bearer authorization. This is not
                the authentication token. You can find the token in
                https://huggingface.co/settings/token. Alternatively, you can
                find both your organizations and personal API tokens using
                `HfApi().whoami(token)`.
            timeout (`int`):
                Timeout in seconds
        """

        headers = build_hf_headers(
            token=token, library_name="text-generation", library_version=__version__
        )

        # Text Generation Inference client only supports a subset of the available hub models
        if not check_model_support(repo_id, headers):
            raise NotSupportedError(repo_id)

        base_url = f"{INFERENCE_ENDPOINT}/models/{repo_id}"

        super(InferenceAPIClient, self).__init__(
            base_url, headers=headers, timeout=timeout
        )


class InferenceAPIAsyncClient(AsyncClient):
    """Aynschronous Client to make calls to the HuggingFace Inference API.

     Only supports a subset of the available text-generation or text2text-generation models that are served using
     text-generation-inference

     Example:

     ```python
     >>> from text_generation import InferenceAPIAsyncClient

     >>> client = InferenceAPIAsyncClient("bigscience/bloomz")
     >>> response = await client.generate("Why is the sky blue?")
     >>> response.generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> async for response in client.generate_stream("Why is the sky blue?"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(self, repo_id: str, token: Optional[str] = None, timeout: int = 10):
        """
        Init headers and API information

        Args:
            repo_id (`str`):
                Id of repository (e.g. `bigscience/bloom`).
            token (`str`, `optional`):
                The API token to use as HTTP bearer authorization. This is not
                the authentication token. You can find the token in
                https://huggingface.co/settings/token. Alternatively, you can
                find both your organizations and personal API tokens using
                `HfApi().whoami(token)`.
            timeout (`int`):
                Timeout in seconds
        """
        headers = build_hf_headers(
            token=token, library_name="text-generation", library_version=__version__
        )

        # Text Generation Inference client only supports a subset of the available hub models
        if not check_model_support(repo_id, headers):
            raise NotSupportedError(repo_id)

        base_url = f"{INFERENCE_ENDPOINT}/models/{repo_id}"

        super(InferenceAPIAsyncClient, self).__init__(
            base_url, headers=headers, timeout=timeout
        )
