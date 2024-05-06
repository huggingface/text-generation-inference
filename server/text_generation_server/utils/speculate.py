SPECULATE = None


def get_speculate() -> int:
    global SPECULATE
    return SPECULATE


def set_speculate(speculate: int):
    global SPECULATE
    SPECULATE = speculate
