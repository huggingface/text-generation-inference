#!/bin/bash

if [[ -z "${HF_MODEL_ID}" ]]; then
  echo "HF_MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
  export REVISION="${HF_MODEL_REVISION}"
fi

if [[ -n "${SM_NUM_GPUS}" ]]; then
  export NUM_SHARD="${SM_NUM_GPUS}"
fi

if [[ -n "${HF_MODEL_QUANTIZE}" ]]; then
  export QUANTIZE="${HF_MODEL_QUANTIZE}"
fi

if [[ -n "${HF_MODEL_TRUST_REMOTE_CODE}" ]]; then
  export TRUST_REMOTE_CODE="${HF_MODEL_TRUST_REMOTE_CODE}"
fi

text-generation-launcher --port 8080
