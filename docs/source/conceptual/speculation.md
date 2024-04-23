## Speculation


Speculative decoding, assisted generation, Medusa, and others are a few different names for the same idea.
The idea is to generate tokens *before* the large model actually runs, and only *check* if those tokens where valid.

So you are making *more* computations on your LLM, but if you are correct you produce 1, 2, 3 etc.. tokens on a single LLM pass. Since LLMs are usually memory bound (and not compute bound), provided your guesses are correct enough, this is a 2-3x faster inference (It can be much more for code oriented tasks for instance).

You can check a more [detailed explanation](https://huggingface.co/blog/assisted-generation).

Text-generation inference supports 2 main speculative methods:

- Medusa
- N-gram


### Medusa


Medusa is a [simple method](https://arxiv.org/abs/2401.10774) to create many tokens in a single pass using fine-tuned LM heads in addition to your existing models.


You can check a few existing  fine-tunes for popular models:

- [text-generation-inference/gemma-7b-it-medusa](https://huggingface.co/text-generation-inference/gemma-7b-it-medusa)
- [text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa](https://huggingface.co/text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa)
- [text-generation-inference/Mistral-7B-Instruct-v0.2-medusa](https://huggingface.co/text-generation-inference/Mistral-7B-Instruct-v0.2-medusa)


In order to create your own medusa heads for your own finetune, you should check own the original medusa repo. [https://github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)


In order to use medusa models in TGI, simply point to a medusa enabled model, and everything will load automatically.


### N-gram


If you don't have a medusa model, or don't have the resource to fine-tune, you can try to use `n-gram`.
N-gram works by trying to find matching tokens in the previous sequence, and use those as speculation for generating new tokens. For example, if the tokens "np.mean" appear multiple times in the sequence, the model can speculate that the next continuation of the tokens "np." is probably also "mean".

This is an extremely simple method, which works best for code, or highly repetitive text. This might not be beneficial, if the speculation misses too much.


In order to enable n-gram speculation simply use

`--speculate 2` in your flags.

[Details about the flag](https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher#speculate)
