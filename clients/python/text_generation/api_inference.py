import os
import requests
import base64
import json
import warnings

from typing import List, Optional

from text_generation import Client, AsyncClient
from text_generation.errors import NotSupportedError

INFERENCE_ENDPOINT = os.environ.get(
    "HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co"
)

SUPPORTED_MODELS = None


def get_supported_models() -> Optional[List[str]]:
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


class APIInferenceClient(Client):
    def __init__(self, model_id: str, token: Optional[str] = None, timeout: int = 10):
        # Text Generation Inference client only supports a subset of the available hub models
        supported_models = get_supported_models()
        if supported_models is not None and model_id not in supported_models:
            raise NotSupportedError(model_id)

        headers = {}
        if token is not None:
            headers = {"Authorization": f"Bearer {token}"}
        base_url = f"{INFERENCE_ENDPOINT}/models/{model_id}"

        super(APIInferenceClient, self).__init__(base_url, headers, timeout)


class APIInferenceAsyncClient(AsyncClient):
    def __init__(self, model_id: str, token: Optional[str] = None, timeout: int = 10):
        # Text Generation Inference client only supports a subset of the available hub models
        supported_models = get_supported_models()
        if supported_models is not None and model_id not in supported_models:
            raise NotSupportedError(model_id)

        headers = {}
        if token is not None:
            headers = {"Authorization": f"Bearer {token}"}
        base_url = f"{INFERENCE_ENDPOINT}/models/{model_id}"

        super(APIInferenceAsyncClient, self).__init__(base_url, headers, timeout)
