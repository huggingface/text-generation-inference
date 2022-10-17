#!/usr/bin/env bash

server_cmd="bloom-inference-server launcher $MODEL_NAME --num-gpus $NUM_GPUS --shard-directory $MODEL_BASE_PATH"

# Run in background
$server_cmd 2>&1 > /dev/null &

# Check if server is running by checking if the unix socket is created
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

# Run in background
text-generation-router &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?