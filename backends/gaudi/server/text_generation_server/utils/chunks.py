from typing import Iterable

from loguru import logger

from text_generation_server.pb import generate_pb2


def concat_text_chunks(chunks: Iterable[generate_pb2.InputChunk]) -> str:
    """
    Concatenate text in text chunks. Non-text chunks are dropped.
    """
    text = None
    for chunk in chunks:
        chunk_type = chunk.WhichOneof("chunk")
        if chunk_type == "text":
            if text is None:
                text = chunk.text
            else:
                raise NotImplementedError("Request contained more than one text chunk")
        else:
            # We cannot reject this, e.g. warmup sends an image chunk.
            logger.debug(f"Encountered non-text chunk type {chunk_type}")

    if text is None:
        raise NotImplementedError("Request without a text chunk")

    return text
