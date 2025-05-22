#!/bin/bash

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

# Check if --sharded argument is present in the command line arguments
if [[ "$*" == *"--sharded true"* ]]; then
  echo 'setting PT_HPU_ENABLE_LAZY_COLLECTIVES=1 for sharding'
  export PT_HPU_ENABLE_LAZY_COLLECTIVES=1
fi
# Check if ATTENTION environment variable is set to paged
if [[ "$ATTENTION" == "paged" ]]; then
  # Check if Llama-4 is in the command line arguments
  if [[ "$*" == *"Llama-4"* || "$*" == *"Qwen3"* ]]; then
    echo 'ATTENTION=paged and Llama-4 or Qwen3 detected'
    pip install transformers==4.52.1
  fi
fi

text-generation-launcher $@
