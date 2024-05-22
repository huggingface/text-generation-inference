import subprocess
import argparse
import ast

TEMPLATE = """
# Supported Models and Hardware

Text Generation Inference enables serving optimized models on specific hardware for the highest performance. The following sections list which models are hardware are supported.

## Supported Models

SUPPORTED_MODELS

If the above list lacks the model you would like to serve, depending on the model's pipeline type, you can try to initialize and serve the model anyways to see how well it performs, but performance isn't guaranteed for non-optimized models:

```python
# for causal LMs/text-generation models
AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`
# or, for text-to-text generation models
AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")
```

If you wish to serve a supported model that already exists on a local folder, just point to the local folder.

```bash
text-generation-launcher --model-id <PATH-TO-LOCAL-BLOOM>
``````


## Supported Hardware

TGI optimized models are supported on NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/), [A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) and [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs with CUDA 12.2+. Note that you have to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to use it. For other NVIDIA GPUs, continuous batching will still apply, but some operations like flash attention and paged attention will not be executed.

TGI also has support of ROCm-enabled AMD Instinct MI210 and MI250 GPUs, with paged attention, GPTQ quantization, flash attention v2 support. The following features are currently not supported in the ROCm version of TGI, and the supported may be extended in the future:
* Loading [AWQ](https://huggingface.co/docs/transformers/quantization#awq) checkpoints.
* Flash [layer norm kernel](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm)
* Kernel for sliding window attention (Mistral)

TGI is also supported on the following AI hardware accelerators:
- *Habana first-gen Gaudi and Gaudi2:* check out this [repository](https://github.com/huggingface/tgi-gaudi) to serve models with TGI on Gaudi and Gaudi2 with [Optimum Habana](https://huggingface.co/docs/optimum/habana/index)
* *AWS Inferentia2:* check out this [guide](https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference) on how to serve models with TGI on Inferentia2.
"""


def check_cli(check: bool):
    output = subprocess.check_output(["text-generation-launcher", "--help"]).decode(
        "utf-8"
    )

    wrap_code_blocks_flag = "<!-- WRAP CODE BLOCKS -->"
    final_doc = f"# Text-generation-launcher arguments\n\n{wrap_code_blocks_flag}\n\n"

    lines = output.split("\n")

    header = ""
    block = []
    for line in lines:
        if line.startswith("  -") or line.startswith("      -"):
            rendered_block = "\n".join(block)
            if header:
                final_doc += f"## {header}\n```shell\n{rendered_block}\n```\n"
            else:
                final_doc += f"```shell\n{rendered_block}\n```\n"
            block = []
            tokens = line.split("<")
            if len(tokens) > 1:
                header = tokens[-1][:-1]
            else:
                header = line.split("--")[-1]
            header = header.upper().replace("-", "_")

        block.append(line)

    rendered_block = "\n".join(block)
    final_doc += f"## {header}\n```shell\n{rendered_block}\n```\n"
    block = []

    filename = "docs/source/basic_tutorials/launcher.md"
    if check:
        with open(filename, "r") as f:
            doc = f.read()
            if doc != final_doc:
                tmp = "launcher.md"
                with open(tmp, "w") as g:
                    g.write(final_doc)
                diff = subprocess.run(
                    ["diff", tmp, filename], capture_output=True
                ).stdout.decode("utf-8")
                print(diff)
                raise Exception(
                    "Cli arguments Doc is not up-to-date, run `python update_doc.py` in order to update it"
                )
    else:
        with open(filename, "w") as f:
            f.write(final_doc)


def check_supported_models(check: bool):
    filename = "server/text_generation_server/models/__init__.py"
    with open(filename, "r") as f:
        tree = ast.parse(f.read())

    enum_def = [
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ModelType"
    ][0]
    _locals = {}
    _globals = {}
    exec(f"import enum\n{ast.unparse(enum_def)}", _globals, _locals)
    ModelType = _locals["ModelType"]
    list_string = ""
    for data in ModelType:
        list_string += f"- [{data.value['name']}]({data.value['url']})"
        if data.value.get("multimodal", None):
            list_string += " (Multimodal)"
        list_string += "\n"

    final_doc = TEMPLATE.replace("SUPPORTED_MODELS", list_string)

    filename = "docs/source/supported_models.md"
    if check:
        with open(filename, "r") as f:
            doc = f.read()
            if doc != final_doc:
                tmp = "launcher.md"
                with open(tmp, "w") as g:
                    g.write(final_doc)
                diff = subprocess.run(
                    ["diff", tmp, filename], capture_output=True
                ).stdout.decode("utf-8")
                print(diff)
                raise Exception(
                    "Supported models is not up-to-date, run `python update_doc.py` in order to update it"
                )
    else:
        with open(filename, "w") as f:
            f.write(final_doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    # check_cli(args.check)
    check_supported_models(args.check)


if __name__ == "__main__":
    main()
