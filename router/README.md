# Router

Also named `webserver` throughout the docs.

This router is handling most of the logic to handle the "batches" tell
when to pass new `prefill` requests and pausing `decode` requests, which ones etc...

It uses gRPC to communicate with the shards which can therefore be kept
much simpler and focus on having the most efficient forward passes as possible.

## Continuous batching

One important feature of `text-generation-inference` is enabled
by this `router`.

Continuous batching is the act of regularly running queries in the same
`forward` step of the LLM (a "batch") and also removing them when they are
finished.

In order for continuous batching to be useful, you need to have more compute available
with respect to the memory requirements of your model. This is essentially true for
LLMs and the larger the model, the truer it gets (since you have to pool multiple
GPUs to load the model, you effectively have a lot of compute power at your hands).


Static batching is the act of doing several queries at the same time, but usually
this is controlled by the client, and therefore the amount of batching is decided
beforehand.

For text-generation, and LLMs which are memory bound we can try to be much more
efficient with the available compute, by having client sending us single queries,
and let the router mix&match queries into or out of batches to make the use the
compute the most efficiently. This is possible because for LLMs the total compute
for running the model is much bigger than doing mix&match of the batches themselves.


### Simple continuous batching

text-generation works by feeding a prompt to a model, and iteratively calling
`forward` on the model to produce new text, 1 token at a time.

The first idea is simple, when a query arrives, we start working on it directly.
When new queries arrive, we simply wait for the current `forward` to be finished
then batch the current running prompt with the new query, and call `forward`.

Whenever either query is finished: either the model produce EOS (end of sentence) token
or the query reached the allowed limit. We simply drop it from the batch, remove
all the allocated memory and we can continue with the rest until nothing is left.

This simple idea generalizes very well and we could potentially stack many requests
in the same batch.

One thing to note, is that queries can be potentially run with different parameters
meaning different way to choose the next token (sampling, not sampling, temperature, top_k etc..). This is not problematic for the proposed approach we just need to do the sampling
independantly on each member of the batch.

### Prefill, decode and past key values

In order to make LLMs and text-generation efficient, there's actually a very powerful
trick that can be used, which is the "caching" of some attention matrices. [More on that
in the first part of this blog](https://huggingface.co/blog/accelerated-inference#getting-to-the-first-10x-speedup)

What this means, is that the first "pass" of a prompt is different from the subsequent
"forward" passes. Since for the first one we have to compute the entire attention matrix, whereas in the follow-ups only require to compute the new token attention.
The first pass is called `prefill` throughout this codebase where as the follow-ups are called `decode`.

Since `prefill` is much more expensive than `decode` we don't want to do it all the time,
but a currently running query is probably doing `decode`. If we want to do the continuous
batching as explained previously we need to run `prefill` at some point in order to create
the attention matrix required to be able to join the `decode` group.

`text-generation-inference` uses a bunch of different strategies and parameters in
order to enable you to find the sweet spot between exploiting the hardware and perceived latency.

With no continuous batching at all, latency is going to be super good, but throughput (meaning
the total number of requests allowed in a given timeframe) is going to be super bad (since it's essentially 1).

With static batching, you can probably reach the maximum throughput (by using the maximum total batch size applicable to your hardware), but the latency is super bad since in order to have maximum throughput you need to wait for requests to come in before processing.

With continuous batching you can find a sweet spot. In general latency is the most critical
parameter users care about. But a 2x latency slowdown for 10x more users on the same
hardware is an acceptable tradeoff.

## Token streaming

This is a very important aspect of client UX. As mentionned above, latency is the
most critical perceived quality of an LLM API.

With token streaming, the server can start answering after the first `prefill` pass
directly, without waiting for all the generation to be done. For extremely long queries
this means clients can start to see something happening orders of magnitude before
the work is done. Seeing something in progress allows them to cut short if it's not
what's wanted but also it "feels" better.
