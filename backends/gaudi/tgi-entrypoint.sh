#!/bin/bash

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

# Check if --sharded argument is present in the command line arguments
if [[ "$*" == *"--sharded true"* ]]; then
  echo 'setting PT_HPU_ENABLE_LAZY_COLLECTIVES=1 for sharding'
  export PT_HPU_ENABLE_LAZY_COLLECTIVES=1
fi

text-generation-launcher $@
