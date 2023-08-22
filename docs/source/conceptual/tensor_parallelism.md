# Tensor Parallelism

Tensor parallelism (also called horizontal model parallelism) is a technique used to fit a large model in multiple GPUs.  Intermediate outputs between ranks are sent and received from one rank to another in a synchronous or asynchronous manner. When multiplying input with weights for inference, multiplying input with weights directly is equivalent to dividing the weight matrix column-wise, multiplying each column with input separately, and then concatenating the separate outputs like below 👇 

![Image courtesy of Anton Lozkhov](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png)

In TGI, tensor parallelism is implemented under the hood by sharding weights and placing them in different ranks. The matrix multiplications then take place in different ranks and are then gathered into a single tensor. 

<Tip warning={true}>

Tensor Parallelism only works for models officially supported, it will not work when falling back on `transformers`.

</Tip>

You can learn more in-depth on tensor-parallelism from transformers docs in this [link](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism).
