from typing import Optional, Tuple, Iterable
from loguru import logger
import json
from text_generation_server.layers.schema import QuantParamSchema


def kv_cache_scales_loader(
    filename: str,
    tp_rank: int,
    tp_size: int,
    num_hidden_layers: int,
    model_type: Optional[str],
) -> Iterable[Tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    previously serialized to disk. Used by the model to populate the appropriate
    KV cache scaling factors. The serialization should represent a dictionary
    whose keys are the TP ranks and values are another dictionary mapping layers
    to their KV cache scaling factors.
    """
    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct, context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()

    except FileNotFoundError:
        logger.error(f"File or directory '{filename}' not found.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in file '{filename}'.")
    except Exception as e:
        logger.error(f"An error occurred while reading '{filename}': {e}")
    # This section is reached if and only if any of the excepts are hit
    # Return an empty iterable (list) => no KV cache scales are loaded
    # which ultimately defaults to 1.0 scales
    logger.warning(
        "Defaulting to KV cache scaling factors = 1.0 "
        f"for all layers in TP rank {tp_rank} "
        "as an error occurred during loading."
    )
    return []
