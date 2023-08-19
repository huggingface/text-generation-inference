# Tensor Parallelism

Tensor Paralellism (also called horizontal model paralellism) is a technique used to fit a large model in multiple GPUs. Model parallelism enables large model training and inference by putting different layers in different GPUs (also called ranks). Intermediate outputs between ranks are sent and received from one rank to another in a synchronous or asynchronous manner. When multiplying input with weights for inference, multiplying input with weights directly is equivalent to dividing weight matrix column-wise, multiplying each column with input separately, and then concatenating the separate outputs like below 👇 

![Image courtesy of Anton Lozkhov](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png)

In TGI, tensor parallelism is implemented under the hood by sharding weights and placing them in different ranks. The matrix multiplications then take place in different ranks and are then gathered into single tensor. 

<Tip warning={true}>

Tensor Parallelism only works for model with custom kernels.

</Tip>
