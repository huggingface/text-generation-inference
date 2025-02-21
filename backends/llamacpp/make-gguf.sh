#!/bin/sh

[ "$#" -ge 2 ] || {
	echo "Usage: $0 <GGUF> <MODEL_ID> [<REV>]" >&2
	return 1
}

case "$1" in (*?.gguf) ;; (*)
	echo "Not a valid GGUF file: $1"
	return 1;
esac

GGUF="$1"
GGUF_DIR=$(dirname -- "$GGUF")
MODEL_ID="$2"
MODEL_DIR="model.src/$2"
REV="${3-main}"

mkdir -p model.src "$GGUF_DIR"

huggingface-cli download \
	--revision "$REV" \
	--local-dir "$MODEL_DIR" \
	"$MODEL_ID" &&

convert_hf_to_gguf.py \
	--outfile "$GGUF" \
	"$MODEL_DIR"

rm -rf -- model.src
