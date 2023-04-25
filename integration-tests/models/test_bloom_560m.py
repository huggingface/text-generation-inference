def test_bloom_560m(bloom_560m, snapshot):
    response = bloom_560m.generate("Test request")
    # response_multi = bloom_560m_multi.generate("Test request")
    # assert response == response_multi == snapshot
    assert response == snapshot

