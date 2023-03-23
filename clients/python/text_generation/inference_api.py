import os
import requests
import base64
import json
import warnings

from typing import List, Optional
from huggingface_hub.utils import build_hf_headers

from text_generation import Client, AsyncClient, __version__
from text_generation.errors import NotSupportedError

INFERENCE_ENDPOINT = os.environ.get(
    "HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co"
)

SUPPORTED_MODELS = None


def get_supported_models() -> Optional[List[str]]:
    """
    Get the list of supported text-generation models from GitHub

    Returns:
        Optional[List[str]]: supported models list or None if unable to get the list from GitHub
    """
    global SUPPORTED_MODELS
    if SUPPORTED_MODELS is not None:
        return SUPPORTED_MODELS

    response = requests.get(
        "https://api.github.com/repos/huggingface/text-generation-inference/contents/supported_models.json",
        timeout=5,
    )
    if response.status_code == 200:
        file_content = response.json()["content"]
        SUPPORTED_MODELS = json.loads(base64.b64decode(file_content).decode("utf-8"))
        return SUPPORTED_MODELS

    warnings.warn("Could not retrieve list of supported models.")
    return None


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

        # Text Generation Inference client only supports a subset of the available hub models
        supported_models = get_supported_models()
        if supported_models is not None and repo_id not in supported_models:
            raise NotSupportedError(repo_id)

        headers = build_hf_headers(
            token=token, library_name="text-generation", library_version=__version__
        )
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

        # Text Generation Inference client only supports a subset of the available hub models
        supported_models = get_supported_models()
        if supported_models is not None and repo_id not in supported_models:
            raise NotSupportedError(repo_id)

        headers = build_hf_headers(
            token=token, library_name="text-generation", library_version=__version__
        )
        base_url = f"{INFERENCE_ENDPOINT}/models/{repo_id}"

        super(InferenceAPIAsyncClient, self).__init__(
            base_url, headers=headers, timeout=timeout
        )
