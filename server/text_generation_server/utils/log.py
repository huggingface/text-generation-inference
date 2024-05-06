from functools import lru_cache


@lru_cache(10)
def log_once(log, msg: str):
    log(msg)
