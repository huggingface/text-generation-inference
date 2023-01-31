import os
import sys
import typer

from pathlib import Path
from loguru import logger
from typing import Optional

from text_generation import server, utils

app = typer.Typer()


@app.command()
def serve(
    model_name: str,
    revision: Optional[str] = None,
    sharded: bool = False,
    quantize: bool = False,
    uds_path: Path = "/tmp/text-generation",
    logger_level: str = "INFO",
    json_output: bool = False,
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )
    if sharded:
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

    server.serve(model_name, revision, sharded, quantize, uds_path)


@app.command()
def download_weights(
    model_name: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
):
    utils.download_weights(model_name, revision, extension)


if __name__ == "__main__":
    app()
