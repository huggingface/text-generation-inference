#!/bin/sh

[ "$#" -ge 2 ] || {
	echo "Usage: $0 <GGUF> <MODEL_ID> [<REV>]" >&2
	return 1
}

GGUF="$1"
GGUF_DIR=$(dirname "$GGUF")
GGUF_TMP="model.src/tmp.gguf"
MODEL_ID="$2"
MODEL_DIR="model.src/$2"
REV="${3-main}"

[ -e "$GGUF" ] && return

mkdir -p model.src "$GGUF_DIR"

huggingface-cli download \
	--revision "$REV" \
	--local-dir "$MODEL_DIR" \
	"$MODEL_ID" &&

convert_hf_to_gguf.py \
	--outfile "$GGUF_TMP" \
	"$MODEL_DIR" &&

llama-quantize \
	"$GGUF_TMP" \
	"$GGUF" \
	"Q4_0"

rm -rf model.src
