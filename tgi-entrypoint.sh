#!/bin/bash

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

source /usr/src/.venv/bin/activate
exec text-generation-launcher $@
