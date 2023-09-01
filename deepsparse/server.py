import fastapi, uvicorn
from contextlib import asynccontextmanager

from threading import Thread
from queue import Queue
from typing import Optional

from router import DeepSparseRouter, batching_task
from utils import GenerateRequestInputs, GenerateRequestOutputs, GenerateRequest

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--deployment-dir", type=str)

args = parser.parse_args()
deployment_dir = args.deployment_dir
model_path = deployment_dir + "/model.onnx"

artifacts = {}

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("\n--------------------       Building Router               --------------------\n")
    artifacts["router"] = DeepSparseRouter(model_path=model_path, tokenizer_path=deployment_dir)

    print("\n--------------------       Starting Batching Task        --------------------\n")
    batching_thread = Thread(target=batching_task, args=[artifacts["router"]])
    batching_thread.start()

    print("\n--------------------       Launching App                 --------------------\n")
    yield
    
    print("\n--------------------       Shutting Down Batching Task   --------------------\n")
    artifacts["router"].stop_batching_task()
    batching_thread.join()

app = fastapi.FastAPI(lifespan=lifespan)

@app.post("/generate")
def generate(inputs: GenerateRequestInputs) -> GenerateRequestOutputs:
    # convert input to generate request
    generate_request = GenerateRequest.from_gr_inputs(inputs)
    
    # submit request to the router
    artifacts["router"].submit_request(generate_request)
    
    gr_outputs = GenerateRequestOutputs()

    # build response
    generation = generate_request.response_stream.get()
    while not generation.stopped:
        gr_outputs.response_text += generation.token
        generation = generate_request.response_stream.get()

    gr_outputs.finish_reason = generation.finish_reason
    return gr_outputs

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5543,
        workers=1,      # limit to one process to avoid copying the model
        # reload=True
    )
