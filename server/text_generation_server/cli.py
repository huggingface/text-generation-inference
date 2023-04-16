import os
import sys
import typer

from pathlib import Path
from loguru import logger
from typing import Optional


app = typer.Typer()


@app.command()
def serve(
    model_id: str,
    revision: Optional[str] = None,
    sharded: bool = False,
    quantize: bool = False,
    uds_path: Path = "/tmp/text-generation-server",
    logger_level: str = "INFO",
    json_output: bool = False,
    otlp_endpoint: Optional[str] = None,
):
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

    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    # Import here after the logger is added to log potential import exceptions
    from text_generation_server import server
    from text_generation_server.tracing import setup_tracing

    # Setup OpenTelemetry distributed tracing
    if otlp_endpoint is not None:
        setup_tracing(shard=os.getenv("RANK", 0), otlp_endpoint=otlp_endpoint)

    server.serve(model_id, revision, sharded, quantize, uds_path)


@app.command()
def download_weights(
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    logger_level: str = "INFO",
    json_output: bool = False,
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    # Import here after the logger is added to log potential import exceptions
    from text_generation_server import utils

    # Test if files were already download
    try:
        utils.weight_files(model_id, revision, extension)
        logger.info(
            "Files are already present in the local cache. " "Skipping download."
        )
        return
    # Local files not found
    except utils.LocalEntryNotFoundError:
        pass

    # Download weights directly
    try:
        filenames = utils.weight_hub_files(model_id, revision, extension)
        utils.download_weights(filenames, model_id, revision)
    except utils.EntryNotFoundError as e:
        if not extension == ".safetensors":
            raise e

        logger.warning(
            f"No safetensors weights found for model {model_id} at revision {revision}. "
            f"Converting PyTorch weights instead."
        )

        # Try to see if there are pytorch weights
        pt_filenames = utils.weight_hub_files(model_id, revision, ".bin")
        # Download pytorch weights
        local_pt_files = utils.download_weights(pt_filenames, model_id, revision)
        local_st_files = [
            p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
            for p in local_pt_files
        ]
        # Convert pytorch weights to safetensors
        utils.convert_files(local_pt_files, local_st_files)


if __name__ == "__main__":
    app()
