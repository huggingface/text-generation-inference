# Neuron backend for AWS Trainium and Inferentia

The Neuron backend allows the deployment of TGI on AWS Trainium and Inferentia family of chips.

The following hardware targets are supported:
- Trainium 1,
- Inferentia 2.

## Features

The basic TGI features are supported:

- continuous batching,
- token streaming,
- greedy search and multinomial sampling using [transformers](https://huggingface.co/docs/transformers/generation_strategies#customize-text-generation).


## Deploy the service from the Hugging Face hub

The simplest way to deploy the NeuronX TGI service for a specific model is to follow the
deployment instructions in the model card:

- click on the "Deploy" button on the right,
- select your deployment service ("Inference Endpoints" and "SageMaker" are supported),
- select "AWS Trainum & Inferentia",
- follow the instructions.


## Deploy the service on a dedicated host

The service is launched simply by running the text-generation-inference container with two sets of parameters:

```
docker run <system_parameters> ghcr.io/huggingface/text-generation-inference:3.2.3-neuron <service_parameters>
```

- system parameters are used to map ports, volumes and devices between the host and the service,
- service parameters are forwarded to the `text-generation-launcher`.

When deploying a service, you will need a pre-compiled Neuron model. The Neuron TGI backend supports two main modes of operation:

- you can either deploy the service on a model that has already been exported to Neuron,
- or alternatively you can take advantage of the Neuron Model Cache to export your own model.

### Common system parameters

Whenever you launch a TGI service, we highly recommend you to mount a shared volume mounted as `/data` in the container: this is where
the models will be cached to speed up further instantiations of the service.

Note also that enough neuron devices should be made visible to the container, knowing that each neuron device has two cores (so when deploying on two cores you need to expose at least one device).
The recommended way to expose a device in a production environment is to use explicitly the `--device` option (e.g `--device /dev/neuron0`) repeated as many time as there are devices to be exposed.

Note: alternatively, for a quick local test it is also possible to launch the service in `privileged` mode to get access to all neuron devices.

Finally, you might want to export the `HF_TOKEN` if you want to access gated repositories.

Here is an example of a service instantiation exposing only the first device:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/text-generation-inference:<VERSION>-neuron \
       <service_parameters>
```

### Using a standard model from the 🤗 [HuggingFace Hub](https://huggingface.co/aws-neuron) (recommended)

We maintain a Neuron Model Cache of the most popular architecture and deployment parameters under [aws-neuron/optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache).

If you just want to try the service quickly using a model without exporting it to Neuron first, it is thus still possible, pending some conditions:
- you must specify the export parameters when launching the service (or use default parameters),
- the model configuration must be cached.

The snippet below shows how you can deploy a service from a hub standard model:

```
export HF_TOKEN=<YOUR_TOKEN>
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       --device=/dev/neuron1 \
       --device=/dev/neuron2 \
       --device=/dev/neuron3 \
       -e HF_TOKEN=${HF_TOKEN} \
       -e HF_AUTO_CAST_TYPE="fp16" \
       -e HF_NUM_CORES=8 \
       ghcr.io/huggingface/text-generation-inference:<VERSION>-neuron \
       --model-id meta-llama/Meta-Llama-3-8B \
       --max-batch-size 1 \
       --max-input-length 3164 \
       --max-total-tokens 4096
```

### Using a model exported to a local path

Alternatively, you can first [export the model to neuron format](https://huggingface.co/docs/optimum-neuron/main/en/guides/export_model#exporting-neuron-models-using-text-generation-inference) locally.

You can then deploy the service inside the shared volume:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       --device=/dev/neuron1 \
       ghcr.io/huggingface/text-generation-inference:<VERSION>-neuron \
       --model-id /data/<neuron_model_path>
```

Note: You don't need to specify any service parameters, as they will all be deduced from the model export configuration. You must however expose enough devices to match the number of cores specified during the export phase.


### Using a neuron model from the 🤗 [HuggingFace Hub](https://huggingface.co/)

The easiest way to share a neuron model inside your organization is to push it on the Hugging Face hub, so that it can be deployed directly without requiring an export.

The snippet below shows how you can deploy a service from a hub neuron model:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       --device=/dev/neuron1 \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/text-generation-inference:<VERSION>-neuron \
       --model-id <organization>/<neuron-model>
```

### Choosing service parameters

Use the following command to list the available service parameters:

```
docker run ghcr.io/huggingface/text-generation-inference:<VERSION>-neuron --help
```

The configuration of an inference endpoint is always a compromise between throughput and latency: serving more requests in parallel will allow a higher throughput, but it will increase the latency.

The neuron models have static input dimensions `[batch_size, max_length]`.

This adds several restrictions to the following parameters:

- `--max-batch-size` must be set to `batch size`,
- `--max-input-length` must be lower than `max_length`,
- `--max-total-tokens` must be set to `max_length` (it is per-request).

Although not strictly necessary, but important for efficient prefilling:

- `--max-batch-prefill-tokens` should be set to `batch_size` * `max-input-length`.

### Choosing the correct batch size

As seen in the previous paragraph, neuron model static batch size has a direct influence on the endpoint latency and throughput.

Please refer to [text-generation-inference](https://github.com/huggingface/text-generation-inference) for optimization hints.

Note that the main constraint is to be able to fit the model for the specified `batch_size` within the total device memory available
on your instance (16GB per neuron core, with 2 cores per device).

## Query the service

You can query the model using either the `/generate` or `/generate_stream` routes:

```
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

Note: replace 127.0.0.1:8080 with your actual IP address and port.
