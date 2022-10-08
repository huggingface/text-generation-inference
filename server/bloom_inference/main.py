import typer

from pathlib import Path
from torch.distributed.launcher import launch_agent, LaunchConfig
from typing import Optional

from bloom_inference.server import serve


def main(
    model_name: str,
    num_gpus: int = 1,
    shard_directory: Optional[Path] = None,
):
    if num_gpus == 1:
        serve(model_name, False, shard_directory)

    else:
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_gpus,
            rdzv_backend="c10d",
            max_restarts=0,
        )
        launch_agent(config, serve, [model_name, True, shard_directory])


if __name__ == "__main__":
    typer.run(main)
