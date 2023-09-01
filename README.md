Install:

```bash
pip3 install deepsparse-nightly[transformers] fastapi uvicorn
```

Download model:
```bash
sparsezoo.download zoo:nlg/text_generation/codegen_multi-350m/pytorch/huggingface/bigquery_thepile/base_quant-none --save-dir codegen-quant
```

Launch server:
```bash
python3 deepsparse/server.py --deployment-dir ./codegen-quant/deployment
```

Make requests:
```python
import requests
from threading import Thread
import json

url = "http://127.0.0.1:5543/generate"
sequence = "Write a function for computing a fibonacci sequence: \n\ndef fib(n):"
# sequence = "def fib(n):"

def request_task(max_new_tokens):
    obj = {
        "inputs": sequence,
        "generation_parameters": {
            "max_new_tokens":max_new_tokens,
            # "repetition_penalty": 1.1,
            # "do_sample": True,
            # "temperature": 1.1,
            # "top_k": 3,
            # "top_p": 0.9,
            # "seed": 42,
        }
    }
    with requests.post(url, json=obj) as r:
        print(max_new_tokens)
        dct = json.loads(r.text)
        # print(dct)
        print(f'{sequence}{dct["response_text"]}')

max_new_tokens_lst = [100, 50, 25]
request_ts = [
    Thread(target=request_task, args=[max_new_tokens]) 
    for max_new_tokens in max_new_tokens_lst
]

import time
for request_t in request_ts:
    request_t.start()
    time.sleep(0.1)

for request_t in request_ts:
    request_t.join()
```