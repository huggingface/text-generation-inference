import fastapi, uvicorn
from contextlib import asynccontextmanager
from threading import Thread
from queue import Queue
from router import DeepSparseRouter, batching_task
from utils import GenerateRequest

TOKENIZER_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/deployment"
MODEL_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/model.onnx/model.onnx"

def serve(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    host="0.0.0.0",
    port=5543
):

    router = None
    
    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        print("\n--------------------       Building Router               --------------------\n")
        router = DeepSparseRouter(
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )

        print("\n--------------------       Starting Batching Task        --------------------\n")
        batching_thread = Thread(target=batching_task, args=[router])
        batching_thread.start()

        print("\n--------------------       Launching App                 --------------------\n")
        yield
        
        print("\n--------------------       Joining Batching Task        --------------------\n")
        router.stop_batching_task()
        batching_task.join()
    
    app = fastapi.FastAPI(lifespan=lifespan)
    
    @app.get("/generate/{prompt}")
    async def generate(prompt:str):
        response_stream = Queue()
        router.submit_request(
            GenerateRequest(
                prompt=prompt,
                max_generated_tokens=100, 
                response_stream=response_stream
            )
        )
        
        response_string = prompt
        generation = response_stream.get()
        while not generation.stopped:
            response_string += generation.token
            generation = response_stream.get()

        return response_string

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1
    )

if __name__ == "__main__":
    serve()