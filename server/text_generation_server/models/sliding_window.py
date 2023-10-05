import os
import math

from typing import Optional

from text_generation_server.models.cache_manager import BLOCK_SIZE

SLIDING_WINDOW: Optional["SlidingWindow"] = None


class SlidingWindow:
    def __init__(self, size: int, attention_sinks: int):
        self.size = size
        self.blocks = math.ceil(size / BLOCK_SIZE)
        self.attention_sinks = attention_sinks

    @classmethod
    def from_env(cls) -> Optional["SlidingWindow"]:
        sliding_window_env = os.getenv("SLIDING_WINDOW", None)
        if sliding_window_env is not None:
            return cls(int(sliding_window_env), int(os.getenv("ATTENTION_SINKS", 0)))
        return None


def set_sliding_window(size: int, attention_sinks: int) -> SlidingWindow:
    global SLIDING_WINDOW
    SLIDING_WINDOW = SlidingWindow(size, attention_sinks)
    return SLIDING_WINDOW


def set_sliding_window_from_env() -> Optional[SlidingWindow]:
    global SLIDING_WINDOW
    env_sliding_window = SlidingWindow.from_env()
    if env_sliding_window is not None:
        SLIDING_WINDOW = env_sliding_window
    return SLIDING_WINDOW


def get_sliding_window() -> Optional[SlidingWindow]:
    global SLIDING_WINDOW
    return SLIDING_WINDOW
