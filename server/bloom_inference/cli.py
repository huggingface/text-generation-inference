import os
import typer

from pathlib import Path
from typing import Optional

from bloom_inference import prepare_weights, server

app = typer.Typer()


@app.command()
def serve(
    model_name: str,
    sharded: bool = False,
    shard_directory: Optional[Path] = None,
    uds_path: Path = "/tmp/bloom-inference",
):
    if sharded:
        assert (
            shard_directory is not None
        ), "shard_directory must be set when sharded is True"
        assert (
            os.getenv("RANK", None) is not None
        ), "RANK must be set when sharded is True"
        assert (
            os.getenv("WORLD_SIZE", None) is not None
        ), "WORLD_SIZE must be set when sharded is True"
        assert (
            os.getenv("MASTER_ADDR", None) is not None
        ), "MASTER_ADDR must be set when sharded is True"
        assert (
            os.getenv("MASTER_PORT", None) is not None
        ), "MASTER_PORT must be set when sharded is True"

    server.serve(model_name, sharded, uds_path, shard_directory)


@app.command()
def prepare_weights(
    model_name: str,
    shard_directory: Path,
    cache_directory: Path,
    num_shard: int = 1,
):
    prepare_weights.prepare_weights(
        model_name, cache_directory, shard_directory, num_shard
    )


if __name__ == "__main__":
    app()
