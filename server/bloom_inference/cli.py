import typer

from pathlib import Path
from torch.distributed.launcher import launch_agent, LaunchConfig
from typing import Optional

from bloom_inference import server

app = typer.Typer()


@app.command()
def launcher(
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
        launch_agent(config, server.serve, [model_name, True, shard_directory])


@app.command()
def serve(
        model_name: str,
        sharded: bool = False,
        shard_directory: Optional[Path] = None,
):
    server.serve(model_name, sharded, shard_directory)


if __name__ == "__main__":
    app()
