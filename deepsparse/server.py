import fastapi, uvicorn
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager

from threading import Thread
from queue import Queue
from typing import Optional

from router import DeepSparseRouter, batching_task
from utils import GenerateRequestInputs, GenerateRequestOutputs, GenerateRequest

TOKENIZER_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/deployment"
MODEL_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/model.onnx/model.onnx"
MESSAGE_STREAM_RETRY_TIMEOUT = 15000  # milisecond

artifacts = {}

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("\n--------------------       Building Router               --------------------\n")
    artifacts["router"] = DeepSparseRouter(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)

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

@app.post("/generate_stream")
async def generate_stream(request: fastapi.Request, inputs: GenerateRequestInputs):
    
    # convert input to generate request
    generate_request = GenerateRequest.from_gr_inputs(inputs)
    
    # submit request to the router
    artifacts["router"].submit_request(generate_request)

    async def token_generator(): 
        while True:
            if await request.is_disconnected():
                break
                
            generation = generate_request.response_stream.get()
            if not generation.stopped:
                yield {
                    "event": "token_generated",
                    "id": "message_id",
                    "retry": MESSAGE_STREAM_RETRY_TIMEOUT,
                    "data": generation.token
                }
            else:
                yield {
                    "event": "token_generated",
                    "id": "message_id",
                    "retry": MESSAGE_STREAM_RETRY_TIMEOUT,
                    "data": generation.finish_reason
                }

    return EventSourceResponse(token_generator())

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5543,
        workers=1,      # limit to one process to avoid copying the model
        # reload=True
    )