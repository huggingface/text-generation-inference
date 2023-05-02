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
    quantize: Optional[str] = None,
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
    auto_convert: bool = True,
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
        logger.info("Files are already present on the host. " "Skipping download.")
        return
    # Local files not found
    except (utils.LocalEntryNotFoundError, FileNotFoundError):
        pass

    is_local_model = (Path(model_id).exists() and Path(model_id).is_dir()) or os.getenv(
        "WEIGHTS_CACHE_OVERRIDE", None
    ) is not None

    if not is_local_model:
        # Try to download weights from the hub
        try:
            filenames = utils.weight_hub_files(model_id, revision, extension)
            utils.download_weights(filenames, model_id, revision)
            # Successfully downloaded weights
            return

        # No weights found on the hub with this extension
        except utils.EntryNotFoundError as e:
            # Check if we want to automatically convert to safetensors or if we can use .bin weights instead
            if not extension == ".safetensors" or not auto_convert:
                raise e

    # Try to see if there are local pytorch weights
    try:
        # Get weights for a local model, a hub cached model and inside the WEIGHTS_CACHE_OVERRIDE
        local_pt_files = utils.weight_files(model_id, revision, ".bin")

    # No local pytorch weights
    except utils.LocalEntryNotFoundError:
        if extension == ".safetensors":
            logger.warning(
                f"No safetensors weights found for model {model_id} at revision {revision}. "
                f"Downloading PyTorch weights."
            )

        # Try to see if there are pytorch weights on the hub
        pt_filenames = utils.weight_hub_files(model_id, revision, ".bin")
        # Download pytorch weights
        local_pt_files = utils.download_weights(pt_filenames, model_id, revision)

    if auto_convert:
        logger.warning(
            f"No safetensors weights found for model {model_id} at revision {revision}. "
            f"Converting PyTorch weights to safetensors."
        )

        # Safetensors final filenames
        local_st_files = [
            p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
            for p in local_pt_files
        ]
        # Convert pytorch weights to safetensors
        utils.convert_files(local_pt_files, local_st_files)


if __name__ == "__main__":
    app()
