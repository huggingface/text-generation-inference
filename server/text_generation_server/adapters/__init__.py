import json
from pathlib import Path
from typing import Dict, Optional

from text_generation_server.adapters.config import AdapterConfig
from text_generation_server.adapters.lora import LoraConfig
from text_generation_server.adapters.weights import (
    AdapterBatchData,
    AdapterBatchMetadata,
)


def load_adapter_config(
    config_path: Optional[Path],
    adapter_config_path: Optional[Path],
    api_token: str,
) -> AdapterConfig:
    if adapter_config_path is not None and adapter_config_path.exists():
        return LoraConfig.load(str(adapter_config_path.parent), api_token)

    raise ValueError(
        f"No valid adapter config file found: "
        f"tried {adapter_config_path} and {config_path}"
    )


__all__ = [
    "AdapterBatchData",
    "AdapterBatchMetadata",
    "load_adapter_config",
]
