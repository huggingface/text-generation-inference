# TGI v3 overview
## Summary


Performance leap: TGI processes 3x more tokens, 13x faster than vLLM on long prompts. Zero config !

### 3x more tokens.
By reducing our memory footprint, we’re able to ingest many more tokens and more dynamically than before. A single L4 (24GB) can handle 30k tokens on llama 3.1-8B, while vLLM gets barely 10k. A lot of work went into reducing the footprint of the runtime and its effect are best seen on smaller constrained environments.

### 13x faster
On long prompts (200k+ tokens) conversation replies take 27.5s in vLLM, while it takes only 2s in TGI. How so ? We keep the initial conversation around, so when a new reply comes in, we can answer almost instantly. The overhead of the lookup is ~5us. Thanks @Daniël de Kok for the beast data structure.

### Zero config
That’s it. Remove all the flags your are using and you’re likely to get the best performance. By evaluating the hardware and model, TGI carefully selects automatic values to give best performance. In production, we don’t have any flags anymore in our deployments. We kept all existing flags around, they may come in handy in niche scenarios.



## Benchmarks

### Methodology

To ensure accurate and reliable results, we employed a robust benchmarking protocol that addresses common pitfalls in performance evaluation. Specifically:

1.  **Consistent Code**: We used the same codebase to run against different engines, ensuring that any performance differences are attributable to the LLM itself, rather than variations in the testing framework.
2.  **Request-Based Measurement**: Instead of measuring Requests Per Second (RPS) by sending as many requests as possible, we opted for a more consistent approach, sending a fixed number of requests and measuring the time it takes for the server to complete all of them. This method avoids boundary effects and provides a more accurate representation of performance.
3.  **Realistic Combinations**: We selected realistic combinations of LLMs and hardware configurations so we used 8xH100 for a 70B, not a 8B, which would be a waste of money.
4.  **Realistic scenarios** We benchmarked engines with prefix caching on, so we are reporting the results of the 2nd run, not the first one.
During the first run of a benchmark, every request is new, so prefix caching is not working, masking the real world benefits of using it.

Note: Boundary effect is when the benchmarks are flaky because their results depend on fine details of the engine being benchmarked.
For instance, a system ingesting a constant 10RPS, but receiving in the benchmark a single final request at -0.1s before the end of the benchmark, and that single request takes a full 10s to process. Then a benchmark taking 30s would measure 7.5RPS instead of the expected 10, because that single query isn't being parallelized with others. Another very slightly slower engine would receive that request at +0.1s which would get discarded by the benchmark and therefore measure the slower system as being faster.

For more details on benchmarking in general we recommend the documentation of k6: https://grafana.com/docs/k6/latest/.

### Scenarios

We selected a handful of scenarios to simplify the picture, they seem to accurately reflect a larger trend.

1. **Small scenario**: This scenario consists of the first 200 requests from the orca datasets being prompted to the model. The 200 requests total 8k tokens together and are representative of conversation starters. Prefix caching has very limited impact in that scenario and we feel it's a relatively balanced benchmark for simple use cases.
2. **Long scenario**: This scenario consists of 20 requests totalling 200k prompt tokens which are essentially asking for summaries of large chunks for text. In practical scenarios this is really useful when you are feeding large chunks of code, large chunks of business data or documents repeatedly and ask simple questions about them (summarization, classification, or where to find some data). This scenario is the one closest to what a lot of professional use cases seem to be doing by including a lot of information in the prompt itself. Those very long conversations are the ones that benefit the most for our recent changes since we are enable ever larger prompts and ever faster caching.

   ### Hardware

   1. `L4` : This is a single L4 (24GB) which represents small or even home compute capabilities. We tested `meta-llama/Meta-Llama-3.1-8B-Instruct` on it.
   2. `4xL4`: This is a more beefy deployment usually used for either very large requests deployments for 8B models (the ones under test) or it can also easily handle all 30GB models. For this benchmark we tested `meta-llama/Meta-Llama-3.1-8B-Instruct`
   3. `8xH100` This is one of the beefiest deployments possible. We tested  `meta-llama/Meta-Llama-3.1-70B-Instruct` as it's the most representative models of this size. Llama 3.3 wasn't released at the time of benchmarking (it's the exact same model so it doesn't make any difference).


### Replicating the results



The commands to run the benchmarks are as follows:

1. Prepare the datasets:

```bash
cd text-generation-inference/load_tests
make prepare_orca
python long.py
```

2. Launch the engine:

TGI: `text-generation-launcher --model-id $MODEL_ID --num-shard $N --port 8000` (or docker variant)
vLLM: `vllm serve $MODEL_ID --tensor-parallel $N —enable-prefix-caching` (or docker variant)

3. Start scenario:
Small: `MODEL_ID=$MODEL_ID  HOST=localhost:8000 k6 run load_tests/common.js`
Long:  `MODEL_ID=$MODEL_ID  HOST=localhost:8000 k6 run load_tests/long.js`


### Results

![benchmarks_v3](https://raw.githubusercontent.com/huggingface/text-generation-inference/refs/heads/main/assets/v3_benchmarks.png)

Our benchmarking results show significant performance gains, with a 13x speedup over vLLM with prefix caching, and up to 30x speedup without prefix caching. These results are consistent with our production data and demonstrate the effectiveness of our optimized LLM architecture.

Raw results

|   |   |   |   |   |
|---|---|---|---|---|
|2nd run ||**TGI v3** (time in s)|**vLLM** (s)|**Amount of req**|
|**Llama 3.1 8b**|Small test - L4 - 8B|17.5|19.9|200|
|**Llama 3.1 8b**|Long test* - L4 - 8B|53|57|10|
|**Llama 3.1 8b**|Small test - 4xL4 - 8B|4.8|6|200|
|**Llama 3.1 8b**|Long test - 4xL4 - 8B|3.2|12.5|20|
|**Llama 3.1 70b**|Small test - 8XH100 - 70B|6.2|7.4|200|
|**Llama 3.1 70b**|Long test - 8H100 - 70B|2|27.5|20|
||||||
|1st run ||TGI (s)|vLLM (s)|Amount of req|
|**Llama 3.1 8b**|Small test - L4|19.9|19.9|200|
|**Llama 3.1 8b**|Long test (10) - L4|49.8|55|10|
|**Llama 3.1 8b**|Small test - 4xL4|13|12.6|200|
|**Llama 3.1 8b**|Long test - 4xL4|47|50.3|20|
|**Llama 3.1 70b**|Small test - 8XH100|7.5|7.6|200|
|**Llama 3.1 70b**|Long test - 8H100|12.1|28.3|20|


### Caveats and Limitations

While our results are promising, there are some caveats to consider:

1. **Constrained kv-cache**: If a deployment lacks kv-cache space, that means that many queries will require the same slots of kv-cache, leading to contention in the kv-cache. You can limit that effect by limiting `--max-total-tokens` to reduce individual queries impact. You can also use more GPUs or larger GPUs in order to increase the size of the kv-cache.
2.  **Replication**: In scenarios where multiple replicas are behind a single endpoint, there's no reason for every query from a particular user to hit the same replica, therefore the cache will not be present, meaning no speed benefit. You can use sticky sessions load balancing to force every user to send their requests on the same replica. Do not apply this blindly, it's possible this may not be necessary at all.

## Technical Insights

Our performance gains can be attributed to several key factors:

1.  **New Kernels**: Our custom kernels, including `flashinfer` and `flashdecoding`, offer improved performance at large prompt lengths and enable more efficient scheduling.
2.  **Prefix Caching**: Our optimized prefix caching structure allows for fast query matching, even for long prompts. The overhead is roughly 6us.
3.  **Chunking Code**: Our chunking code enables finer control over compute resources, ensuring optimal performance and reduced VRAM usage.
4.  **Kernel Optimizations**: We've implemented various other kernel optimizations, including better kernel selection. Notably we've implemented several small kernels involved in the queries bookkeeping which are particularly efficient on small models. Every kernel launch has an overhead of several milliseconds so fusing them together increases a lot performance when this bookkeeping is important relative to the raw model calculations. This happens typically on oversized compute for a particular model and particularly small models.
5. **VRAM efficiency**: In the realm of very large requests (100k+ tokens) there are a lot of places which start becoming big memory consumers. We've hunted the biggest ones and found ways to reduce/reuse or delete them. The biggest culprit probably is `logits` calculation. Logits for llama 3.1-8b take 25.6GB (=100k tokens * 128k vocabulary * 2(f16)) which is more than the full model which is 16GB. The thing is that in general we do not need every prompt logits, so we simply removed them and removed them from being potentially asked by users by default. We think this is ok since they are mostly used by researchers. You can enable your deployments to have them again by using the `--enable-prefill-logprobs` flag, but you will experience reduced token prompt size.

## Future Directions

While we've made significant progress, there are still opportunities for improvement:

1.  **Special models**: All LLMs come with the aforementioned improvements. Some specific set of features might not (some quantizations, speculation or VLMs for instance are harder to optimize for with the same level of detail).
2.  **KV-Cache Long-Term Retention**: Addressing KV-cache long-term retention is a challenge. There are several solutions envisionned like shared KV-cache (like redis or memcached) solutions or innovative storage approaches. It is an area of ongoing research of ours.
3.  **Multimodal models**: We are also investigating quite a lot other kind of models, like audio-to-audio, image/video generation, and other hybrids, where we see a lot of potential of applying the same principles we've applied in TGI to maximize performance.

By sharing our benchmarking methodology, results, and technical insights, we aim to contribute to the ongoing development of more efficient and effective LLMs.
