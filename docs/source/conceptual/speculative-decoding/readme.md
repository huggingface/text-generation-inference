# 🔄🔍 Speculative Decoding 
This project provides a simple implementation of the [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) paper by Leviathan et al. The implementation uses pure NumPy for a basic GPT-2 model, demonstrating the concept of speculative decoding in a straightforward manner.

Key features of this implementation:
- Uses NumPy for all computations, making it easy to understand and modify
- Implements speculative decoding for a GPT-2 model
- Compares performance between standard autoregressive sampling and speculative sampling
- Provides a clear example of how speculative decoding can accelerate language model inference

This simple implementation serves as an educational tool to understand the core concepts of speculative decoding and its potential benefits in accelerating large language model inference.

# Speculative Decoding in a nutshell
Speculative decoding is an innovative technique designed to accelerate the inference process of large language models. Here's a brief overview of how it works:

1. Draft Model: A smaller, faster "draft" model generates a sequence of K tokens quickly.

2. Target Model: The larger, more accurate "target" model processes the entire sequence (input + draft) in parallel.

3. Verification: The target model's output is compared with the draft model's predictions.

4. Accept or Reject: 
   - If the target model agrees with a draft token, it's accepted.
   - If there's a disagreement, the draft is rejected, and the target model's prediction is used instead.

5. Efficiency Gain: This approach allows the target model to process multiple tokens in a single forward pass, potentially reducing the number of expensive computations.

The key advantage is that when the draft model's predictions are mostly correct, the process can be significantly faster than traditional autoregressive decoding. Even when the draft model makes mistakes, the performance doesn't degrade below that of standard autoregressive sampling.

This method leverages the speed of smaller models and the accuracy of larger ones, offering a balance between inference speed and output quality.


# 🚀 How to Use

To run the speculative decoding implementation, use the following command:

```bash
python main.py \
    --prompt "Quantization also improves latency and throughput but suffer from perf" \
    --n_tokens_to_generate 60 \
    --draft_model_size "124M" \
    --target_model_size "355M" \
    --K 4 \
    --temperature 0 # 0 for greedy sampling
```
Sample Output:

```
Autoregressive Decoding
--------------------------------------------------
Time = 112.19s
Text = Quantization also improves latency and throughput but suffer from perfomance issues.

The problem is that the performance of the GPU is not the only thing that matters. The CPU is also important. The CPU is the main bottleneck in the GPU. The CPU is the main bottleneck in the GPU.

The CPU is the main bottleneck in the GPU

Speculative Decoding
--------------------------------------------------
Time = 74.12s
Text = Quantization also improves latency and throughput but suffer from perfomance issues.

The problem is that the performance of the GPU is not the only thing that matters. The CPU is also important. The CPU is the main bottleneck in the GPU. The CPU is the main bottleneck in the GPU.

The CPU is the main bottleneck in the GPU. The CPU

```



# 🤔💭 Why this works?
Most of the work getting done is **NOT** about computation, but its actually about all those read/writes to access memory.
Bc whats happening is that the input lives on the memory and when you do any computation, it has to travel to the GPU/ to all the caches and registers to do the computation and then back to the memory. This is a very slow process. 
![alt text](img/image.png)

So each time we are doing round trips which is slow and very expensive. SO the idea is basically we gonna do a single trip to GPU and while that memory or at least a chunk of it is in the GPU, we are gonna do as much computation as possible and then we gonna load back the results to the memory.

> "Now the clever idea is to use a small and cheap draft model to first generate a candidate sequence of K tokens - a 'draft'. Then we feed all of these together through the big model in a batch. This is almost as fast as feeding in just one token, per the above. Then we go from left to right over the logits predicted by the model and sample tokens. Any sample that agrees with the draft allows us to immediately skip forward to the next token. If there is a disagreement then we throw the draft away and eat the cost of doing some throwaway work (sampling the draft and the forward passing for all the later tokens).
> 
> The reason this works in practice is that most of the time the draft tokens get accepted, because they are easy, so even a much smaller draft model gets them. As these easy tokens get accepted, we skip through those parts in leaps. The hard tokens where the big model disagrees 'fall back' to original speed, but actually a bit slower because of all the extra work."
> 
> — Andrej Karpathy



# 🧮💡Why this works mathematically?

Speculative decoding's mathematical foundation is rooted in rejection sampling, a Monte Carlo method used to generate samples from a draft/smaller distribution when direct sampling from the target/larger distribution is difficult.

## Mathematical Foundation: [Rejection Sampling](https://en.wikipedia.org/wiki/Rejection_sampling)

Speculative decoding's mathematical foundation is rooted in rejection sampling, a Monte Carlo method used to generate samples from a target distribution when direct sampling is difficult. The process involves using a proposal distribution (the draft model) that's easier to sample from, then accepting or rejecting these samples based on comparison with the target distribution (the large model). The rejection sampling theorem guarantees that if we sample from the proposal distribution and accept samples with probability proportional to the ratio of target to proposal distributions, the accepted samples will follow the target distribution exactly. The reason of why this so magically works roots back to the bayes rule that we use to calculate the conditional probability of the next token given the previous context.

## ❌🎯 Rejection Sampling Theorem

The theorem states that if we have a target distribution \( p \) and a proposal distribution \( q \), and we sample from \( q \) and accept samples with probability proportional to the ratio of \( p \) to \( q \), the accepted samples will follow the target distribution \( p \).

Mathematically, this can be expressed as:

1. Sample y from q(y)
2. Accept y with probability min(1, p(y) / (M * q(y)))

Where:
- p(y) is the target distribution
- q(y) is the proposal distribution
- M is a constant such that M ≥ max(p(y) / q(y)) for all y

If we follow this procedure, the accepted samples will be distributed according to p(y).

## Question: What if we dont have access to the same family model for both draft and target model?

Alternative methods like;

- Medusa
- N-gram


### Medusa


Medusa is a [simple method](https://arxiv.org/abs/2401.10774) to create many tokens in a single pass using fine-tuned LM heads in addition to your existing models.


You can check a few existing  fine-tunes for popular models:

- [text-generation-inference/gemma-7b-it-medusa](https://huggingface.co/text-generation-inference/gemma-7b-it-medusa)
- [text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa](https://huggingface.co/text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa)
- [text-generation-inference/Mistral-7B-Instruct-v0.2-medusa](https://huggingface.co/text-generation-inference/Mistral-7B-Instruct-v0.2-medusa)


In order to create your own medusa heads for your own finetune, you should check own the original medusa repo. [../basic_tutorials/train_medusa.md](../basic_tutorials/train_medusa.md)


In order to use medusa models in TGI, simply point to a medusa enabled model, and everything will load automatically.


### N-gram


If you don't have a medusa model, or don't have the resource to fine-tune, you can try to use `n-gram`.
N-gram works by trying to find matching tokens in the previous sequence, and use those as speculation for generating new tokens. For example, if the tokens "np.mean" appear multiple times in the sequence, the model can speculate that the next continuation of the tokens "np." is probably also "mean".

This is an extremely simple method, which works best for code, or highly repetitive text. This might not be beneficial, if the speculation misses too much.


In order to enable n-gram speculation simply use

`--speculate 2` in your flags. [Details about the flag](https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher#speculate)


Please refer to [Speculation](https://huggingface.co/docs/text-generation-inference/conceptual/speculation) for more details.


# ⚡🚀 Summary of most common speed up techniques:
## 🧠💻 Faster Training
- Device: Move on to GPU
- Mix percisions
- Gradient Accumulation
- Distributed Training: 

## ⚡🤖 Faster Inference
- Quantization
- Speculative Decoding (This repo 💖)
- Pruning
- Caching
    - inference-attention: KV cache
    - in production: Prompt cache/ Exact cache/ Semantic cache
- Knowledge Distillation


