# encoding:utf-8
# -------------------------------------------#
# Filename: aime-local-inference-server -- test_llama_hf.py
#
# Description:   
# Version:       1.0
# Created:       2023/9/7-15:19
# Last modified by: 
# Author:        'zhaohuayang@myhexin.com'
# Company:       同花顺网络信息股份有限公司
# -------------------------------------------#
import time

import torch
import transformers


def test_llama():
    tokenizer = transformers.AutoTokenizer.from_pretrained("/code/models/llama-7b-hf")
    model = transformers.AutoModelForCausalLM.from_pretrained("/code/models/llama-7b-hf", torch_dtype=torch.float16,
                                                              device_map="auto").half().eval()
    prompt = "who are you?"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    # warm up
    output_ids = model.generate(input_ids, max_new_tokens=100)
    i_l = len(input_ids[0])
    o_l = len(output_ids[0])
    print(f'input length: {i_l}\t'
          f'output length: {o_l}')
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    # tps
    loop = 10
    s_time = time.time()
    for _ in range(loop):
        model.generate(input_ids, max_new_tokens=100)
    mean_ = (time.time() - s_time) / loop

    print(f"tps: {(o_l - i_l) / mean_:.4f}")


if __name__ == '__main__':
    test_llama()
