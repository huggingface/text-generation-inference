import uvicorn, fastapi
from threading import Thread
from queue import Queue

from router import DeepSparseRouter, batching_task
from utils import GenerateRequest

TOKENIZER_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/deployment"
MODEL_PATH = "/home/robertgshaw/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/model.onnx/model.onnx"

# setup router
router = DeepSparseRouter(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH
)

# start background routing task
batching_thread = Thread(target=batching_task, args=[router])
batching_thread.start()

app = fastapi.FastAPI()

@app.post("/generate")
def generate(prompt:str, max_generated_tokens:int):
    response_stream = Queue()

    # submit request to the router
    router.submit_request(
        generate_request=GenerateRequest(
            prompt=prompt,
            max_generated_tokens=max_generated_tokens,
            response_stream=response_stream
        )
    )
    
    response_string = prompt
    generation = response_stream.get()
    while not generation.stopped:
        response_string += generation.token
        generation = response_stream.get()

    return generation