import torch
from text_generation_server.utils.tokens import (
    StopSequenceCriteria,
    StoppingCriteria,
    FinishReason,
    batch_top_tokens,
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


def test_batch_top_tokens():
    top_n_tokens = [0, 2, 3, 4, 5]
    top_n_tokens_tensor = torch.tensor(top_n_tokens)
    inp_logprobs = torch.tensor([[-1.0, -3.0, -4.0, -2.0, -3.0]] * 5)
    accepted_ids = torch.ones_like(top_n_tokens_tensor)

    topn_tok_ids, topn_tok_logprobs = batch_top_tokens(
        top_n_tokens, top_n_tokens_tensor, inp_logprobs, accepted_ids
    )

    assert topn_tok_ids[0] == [[]]
    assert topn_tok_ids[1] == [[0, 3]]
    assert topn_tok_ids[2] == [[0, 3, 1, 4]]
    assert topn_tok_ids[3] == [[0, 3, 1, 4]]
    assert topn_tok_ids[4] == [[0, 3, 1, 4, 2]]

    assert topn_tok_logprobs[0] == [[]]
    assert topn_tok_logprobs[1] == [[-1, -2]]
    assert topn_tok_logprobs[2] == [[-1, -2, -3, -3]]
    assert topn_tok_logprobs[3] == [[-1, -2, -3, -3]]
    assert topn_tok_logprobs[4] == [[-1, -2, -3, -3, -4]]

    # Now let's make second member of the batch be speculated
    inp_logprobs = torch.tensor([[-1.0, -3.0, -4.0, -2.0, -3.0]] * 5 * 2)
    accepted_ids[1] = 2
    topn_tok_ids, topn_tok_logprobs = batch_top_tokens(
        top_n_tokens, top_n_tokens_tensor, inp_logprobs, accepted_ids
    )

    assert topn_tok_ids[0] == [[]]
    assert topn_tok_ids[1] == [[0, 3], [0, 3]]
    assert topn_tok_ids[2] == [[0, 3, 1, 4]]
    assert topn_tok_ids[3] == [[0, 3, 1, 4]]
    assert topn_tok_ids[4] == [[0, 3, 1, 4, 2]]

    assert topn_tok_logprobs[0] == [[]]
    assert topn_tok_logprobs[1] == [[-1, -2], [-1, -2]]
    assert topn_tok_logprobs[2] == [[-1, -2, -3, -3]]
    assert topn_tok_logprobs[3] == [[-1, -2, -3, -3]]
    assert topn_tok_logprobs[4] == [[-1, -2, -3, -3, -4]]
