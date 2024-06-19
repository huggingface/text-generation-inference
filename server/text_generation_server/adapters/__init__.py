# Origin:   https://github.com/predibase/lorax
# Path:     lorax/server/lorax_server/adapters/__init__.py
# License:  Apache License Version 2.0, January 2004

from text_generation_server.adapters.weights import (
    AdapterBatchData,
    AdapterBatchMetadata,
)

__all__ = [
    "AdapterBatchData",
    "AdapterBatchMetadata",
]
