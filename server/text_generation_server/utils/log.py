from functools import lru_cache
from text_generation_server.utils.dist import RANK


@lru_cache(10)
def log_once(log, msg: str, master=True):
    if master:
        log_master(log, msg)
    else:
        log(msg)


def log_master(log, msg: str):
    if RANK == 0:
        log(msg)
