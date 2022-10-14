#!/usr/bin/env bash

server_cmd="python server/bloom_inference/main.py $MODEL_NAME --num-gpus $NUM_GPUS --shard-directory $MODEL_BASE_PATH"
$server_cmd &

FILE=/tmp/bloom-inference-0

while :
  do
    if test -S "$FILE"; then
        echo "Text Generation Python gRPC server started"
        break
    else
      echo "Waiting for Text Generation Python gRPC server to start"
      sleep 5
    fi
  done

sleep 1

exec "bloom-inference"
