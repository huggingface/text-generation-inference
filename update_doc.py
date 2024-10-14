import subprocess
import argparse
import ast
import json
import os

TEMPLATE = """
# Supported Models

Text Generation Inference enables serving optimized models. The following sections list which models (VLMs & LLMs) are supported.

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
```
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

    filename = "docs/source/reference/launcher.md"
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
                tmp = "supported.md"
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


def get_openapi_schema():
    try:
        output = subprocess.check_output(["text-generation-router", "print-schema"])
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        print(f"Error running text-generation-router print-schema: {e}")
        raise SystemExit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON received from text-generation-router print-schema")
        raise SystemExit(1)


def check_openapi(check: bool):
    new_openapi_data = get_openapi_schema()
    filename = "docs/openapi.json"
    tmp_filename = "openapi_tmp.json"

    with open(tmp_filename, "w") as f:
        json.dump(new_openapi_data, f, indent=2)

    if check:
        diff = subprocess.run(
            [
                "diff",
                # allow for trailing whitespace since it's not significant
                # and the precommit hook will remove it
                "--ignore-trailing-space",
                tmp_filename,
                filename,
            ],
            capture_output=True,
        ).stdout.decode("utf-8")
        os.remove(tmp_filename)

        if diff:
            print(diff)
            raise Exception(
                "OpenAPI documentation is not up-to-date, run `python update_doc.py` in order to update it"
            )

    else:
        os.rename(tmp_filename, filename)
        print("OpenAPI documentation updated.")
    p = subprocess.run(
        [
            "redocly",
            # allow for trailing whitespace since it's not significant
            # and the precommit hook will remove it
            "lint",
            filename,
        ],
        capture_output=True,
    )
    errors = p.stderr.decode("utf-8")
    # The openapi specs fails on `exclusive_minimum` which is expected to be a boolean where
    # utoipa outputs a value instead: https://github.com/juhaku/utoipa/issues/969
    print(errors)
    if p.returncode != 0:
        print(errors)
        raise Exception(
            f"OpenAPI documentation is invalid, `redocly lint {filename}` showed some error:\n {errors}"
        )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    check_cli(args.check)
    check_supported_models(args.check)
    check_openapi(args.check)


if __name__ == "__main__":
    main()
