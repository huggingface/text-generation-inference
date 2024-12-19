# Guidance

## What is Guidance?

Guidance is a feature that allows users to constrain the generation of a large language model with a specified grammar. This feature is particularly useful when you want to generate text that follows a specific structure or uses a specific set of words or produce output in a specific format. A prominent example is JSON grammar, where the model is forced to output valid JSON.

## How is it used?

Guidance can be implemented in many ways and the community is always finding new ways to use it. Here are some examples of how you can use guidance:

Technically, guidance can be used to generate:

- a specific JSON object
- a function signature
- typed output like a list of integers

However these use cases can span a wide range of applications, such as:

- extracting structured data from unstructured text
- summarizing text into a specific format
- limit output to specific classes of words (act as a LLM powered classifier)
- generate the input to specific APIs or services
- provide reliable and consistent output for downstream tasks
- extract data from multimodal inputs

## How it works?

Diving into the details, guidance is enabled by including a grammar with a generation request that is compiled, and used to modify the chosen tokens.

This process can be broken down into the following steps:

1. A request is sent to the backend, it is processed and placed in batch. Processing includes compiling the grammar into a finite state machine and a grammar state.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/request-to-batch.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/request-to-batch-dark.gif"
    />
</div>

2. The model does a forward pass over the batch. This returns probabilities for each token in the vocabulary for each request in the batch.

3. The process of choosing one of those tokens is called `sampling`. The model samples from the distribution of probabilities to choose the next token. In TGI all of the steps before sampling are called `processor`. Grammars are applied as a processor that masks out tokens that are not allowed by the grammar.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/logit-grammar-mask.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/logit-grammar-mask-dark.gif"
    />
</div>

4. The grammar mask is applied and the model samples from the remaining tokens. Once a token is chosen, we update the grammar state with the new token, to prepare it for the next pass.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/sample-logits.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/sample-logits-dark.gif"
    />
</div>

## How to use Guidance?

There are two main ways to use guidance; you can either use the `/generate` endpoint with a grammar or use the `/chat/completion` endpoint with tools.

Under the hood tools are a special case of grammars that allows the model to choose one or none of the provided tools.

Please refer to [using guidance](../basic_tutorials/using_guidance) for more examples and details on how to use guidance in Python, JavaScript, and cURL.

### Getting the most out of guidance

Depending on how you are using guidance, you may want to make use of different features. Here are some tips to get the most out of guidance:

- If you are using the `/generate` with a `grammar` it is recommended to include the grammar in the prompt prefixed by something like `Please use the following JSON schema to generate the output:`. This will help the model understand the context of the grammar and generate the output accordingly.
- If you are getting a response with many repeated tokens, please use the `frequency_penalty` or `repetition_penalty` to reduce the number of repeated tokens in the output.
