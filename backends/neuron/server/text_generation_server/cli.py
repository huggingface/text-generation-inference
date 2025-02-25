import sys
from typing import Optional

import typer
from loguru import logger


app = typer.Typer()


@app.command()
def serve(
    model_id: str,
    revision: Optional[str] = None,
    sharded: bool = False,
    trust_remote_code: bool = None,
    uds_path: str = "/tmp/text-generation-server",
    logger_level: str = "INFO",
    json_output: bool = False,
    otlp_endpoint: Optional[str] = None,
    otlp_service_name: str = "text-generation-inference.server",
    max_input_tokens: Optional[int] = None,
):
    """This is the main entry-point for the server CLI.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.
        sharded (`bool`):
            Whether the model must be sharded or not. Kept for compatibility with the
            text-generation-launcher, but must be set to False.
        trust-remote-code (`bool`):
            Kept for compatibility with text-generation-launcher. Ignored.
        uds_path (`Union[Path, str]`):
            The local path on which the server will expose its google RPC services.
        logger_level (`str`):
            The server logger level. Defaults to *INFO*.
        json_output (`bool`):
            Use JSON format for log serialization.
        otlp_endpoint (`Optional[str]`, defaults to `None`):
            The Open Telemetry endpoint to use.
        otlp_service_name (`Optional[str]`, defaults to `None`):
            The name to use when pushing data to the Open Telemetry endpoint.
        max_input_tokens (`Optional[int]`, defaults to `None`):
            The maximum number of input tokens each request should contain.
    """
    if sharded:
        raise ValueError("Sharding is not supported.")
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

    if trust_remote_code is not None:
        logger.warning(
            "'trust_remote_code' argument is not supported and will be ignored."
        )

    # Import here after the logger is added to log potential import exceptions
    from .server import serve

    serve(model_id, revision, uds_path)


@app.command()
def download_weights(
    model_id: str,
    revision: Optional[str] = None,
    logger_level: str = "INFO",
    json_output: bool = False,
    auto_convert: Optional[bool] = None,
    extension: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    merge_lora: Optional[bool] = None,
):
    """Download the model weights.

    This command will be called by text-generation-launcher before serving the model.
    """
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

    if extension is not None:
        logger.warning("'extension' argument is not supported and will be ignored.")
    if trust_remote_code is not None:
        logger.warning(
            "'trust_remote_code' argument is not supported and will be ignored."
        )
    if auto_convert is not None:
        logger.warning("'auto_convert' argument is not supported and will be ignored.")
    if merge_lora is not None:
        logger.warning("'merge_lora' argument is not supported and will be ignored.")

    # Import here after the logger is added to log potential import exceptions
    from .model import fetch_model

    fetch_model(model_id, revision)
