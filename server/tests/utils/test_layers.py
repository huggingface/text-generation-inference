import torch
from text_generation_server.layers import (
    TensorParallelEmbedding,
)


class ProcessGroup:
    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self.world_size = world_size

    def size(self) -> int:
        return self.world_size

    def rank(self) -> int:
        return self._rank


class Weights:
    def __init__(self, rank: int, world_size: int, vocab_size: int, hidden_dim: int):
        self.weight = (
            torch.arange(vocab_size * hidden_dim).float().view(vocab_size, hidden_dim)
        )
        self.process_group = ProcessGroup(rank, world_size)

    def get_partial_sharded(self, name: str, dim: int):
        assert dim == 0

        rank = self.process_group.rank()
        world_size = self.process_group.size()
        size = self.weight.shape[dim]

        block_size = (size + world_size - 1) // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
        return self.weight[start:stop]

    def get_shape(self, name: str):
        return self.weight.shape


def test_weight_hub_files_offline_error():

    vocab_size = 17
    weights = Weights(rank=0, world_size=1, vocab_size=vocab_size, hidden_dim=256)
    embeddings = TensorParallelEmbedding("", weights)

    input_ids = torch.arange(vocab_size)
    output = embeddings.forward(input_ids)
    assert embeddings.min_id == 0
    assert embeddings.max_id == 17
    torch.testing.assert_close(output, torch.arange(256 * 17).float().view(17, 256))

    weights_0_2 = Weights(rank=0, world_size=2, vocab_size=vocab_size, hidden_dim=256)
    weights_1_2 = Weights(rank=1, world_size=2, vocab_size=vocab_size, hidden_dim=256)
    embeddings_0_2 = TensorParallelEmbedding("", weights_0_2, reduce=False)
    assert embeddings_0_2.min_id == 0
    assert embeddings_0_2.max_id == 9
    torch.testing.assert_close(
        embeddings_0_2.weight,
        torch.cat([torch.arange(9 * 256), torch.zeros(256)], dim=0)
        .view(10, 256)
        .float(),
    )
    embeddings_1_2 = TensorParallelEmbedding("", weights_1_2, reduce=False)
    assert embeddings_1_2.min_id == 9
    assert embeddings_1_2.max_id == 17
    torch.testing.assert_close(
        embeddings_1_2.weight,
        torch.cat([torch.arange(8 * 256) + 9 * 256, torch.zeros(256)], dim=0)
        .view(9, 256)
        .float(),
    )
    output_tp_0 = embeddings_0_2.forward(input_ids)
    output_tp_1 = embeddings_1_2.forward(input_ids)

    torch.testing.assert_close(output, output_tp_0 + output_tp_1)
