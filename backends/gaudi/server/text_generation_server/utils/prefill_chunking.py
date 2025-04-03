from typing import Optional

SUPPORT_CHUNKING: Optional[bool] = None
MAX_PREFILL_TOKENS: Optional[int] = None


def set_support_chunking(support_chunking: bool):
    global SUPPORT_CHUNKING
    SUPPORT_CHUNKING = support_chunking


def get_support_chunking() -> bool:
    global SUPPORT_CHUNKING
    return SUPPORT_CHUNKING


def set_max_prefill_tokens(max_prefill_tokens: int):
    global MAX_PREFILL_TOKENS
    MAX_PREFILL_TOKENS = max_prefill_tokens


def get_max_prefill_tokens() -> int:
    global MAX_PREFILL_TOKENS
    return MAX_PREFILL_TOKENS
