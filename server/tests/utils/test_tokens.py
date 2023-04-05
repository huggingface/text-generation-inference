from text_generation_server.utils.tokens import (
    StopSequenceCriteria,
    StoppingCriteria,
    FinishReason,
)


def test_stop_sequence_criteria():
    criteria = StopSequenceCriteria("/test;")

    assert not criteria("/")
    assert not criteria("/test")
    assert criteria("/test;")
    assert not criteria("/test; ")


def test_stop_sequence_criteria_escape():
    criteria = StopSequenceCriteria("<|stop|>")

    assert not criteria("<")
    assert not criteria("<|stop")
    assert criteria("<|stop|>")
    assert not criteria("<|stop|> ")


def test_stopping_criteria():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(65827, "/test") == (False, None)
    assert criteria(30, ";") == (True, FinishReason.FINISH_REASON_STOP_SEQUENCE)


def test_stopping_criteria_eos():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(0, "") == (True, FinishReason.FINISH_REASON_EOS_TOKEN)


def test_stopping_criteria_max():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (True, FinishReason.FINISH_REASON_LENGTH)
