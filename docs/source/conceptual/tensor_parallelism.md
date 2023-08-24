# Tensor Parallelism

Tensor parallelism is a technique used to fit a large model in multiple GPUs. For example, when multiplying the input tensors with the first weight tensor, multiplying both tensors is equivalent to splitting the weight tensor column-wise, multiplying each column with input separately, and then concatenating the separate outputs. These outputs are then sent between GPUs and then concatenated together to get the final result, like below 👇 

![Image courtesy of Anton Lozkhov](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png)


<Tip warning={true}>

Tensor Parallelism only works for models officially supported, it will not work when falling back on `transformers`. You can get more information about unsupported models [here](./basic_tutorials/non_core_models.md).

</Tip>

You can learn more in-depth about tensor-parallelism from `transformers` docs in this [link](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism).
