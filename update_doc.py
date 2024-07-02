import subprocess
import argparse
import ast
import requests
import json
import time
import os

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


def start_server_and_wait():
    log_file = open("/tmp/server_log.txt", "w")

    process = subprocess.Popen(
        ["text-generation-launcher"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    print("Server is starting...")

    start_time = time.time()
    while True:
        try:
            response = requests.get("http://127.0.0.1:3000/health")
            if response.status_code == 200:
                print("Server is up and running!")
                return process, log_file
        except requests.RequestException:
            if time.time() - start_time > 60:
                log_file.close()
                with open("server_log.txt", "r") as f:
                    print("Server log:")
                    print(f.read())
                os.remove("server_log.txt")
                raise TimeoutError("Server didn't start within 60 seconds")
            time.sleep(1)


def stop_server(process, log_file, show=False):
    process.terminate()
    process.wait()
    log_file.close()

    if show:
        with open("/tmp/server_log.txt", "r") as f:
            print("Server log:")
            print(f.read())
    os.remove("/tmp/server_log.txt")


def get_openapi_json():
    response = requests.get("http://127.0.0.1:3000/api-doc/openapi.json")
    # error if not 200
    response.raise_for_status()
    return response.json()


def update_openapi_json(new_data, filename="docs/openapi.json"):
    with open(filename, "w") as f:
        json.dump(new_data, f, indent=2)


def compare_openapi(old_data, new_data):
    differences = []

    def compare_recursive(old, new, path=""):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in set(old.keys()) | set(new.keys()):
                new_path = f"{path}.{key}" if path else key
                if key not in old:
                    differences.append(f"Added: {new_path}")
                elif key not in new:
                    differences.append(f"Removed: {new_path}")
                else:
                    compare_recursive(old[key], new[key], new_path)
        elif old != new:
            differences.append(f"Changed: {path}")

    compare_recursive(old_data, new_data)
    return differences


def openapi(check: bool):
    try:
        server_process, log_file = start_server_and_wait()

        try:
            new_openapi_data = get_openapi_json()

            if check:
                try:
                    with open("docs/openapi.json", "r") as f:
                        old_openapi_data = json.load(f)
                except FileNotFoundError:
                    print(
                        "docs/openapi.json not found. Run without --check to create it."
                    )
                    return

                differences = compare_openapi(old_openapi_data, new_openapi_data)

                if differences:
                    print("The following differences were found:")
                    for diff in differences:
                        print(diff)
                    print(
                        "Please run the script without --check to update the documentation."
                    )
                else:
                    print("Documentation is up to date.")
            else:
                update_openapi_json(new_openapi_data)
                print("Documentation updated successfully.")

        finally:
            stop_server(server_process, log_file)

    except TimeoutError as e:
        print(f"Error: {e}")
    except requests.RequestException as e:
        print(f"Error communicating with the server: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON received from the server")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update documentation for text-generation-launcher"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    openapi_parser = subparsers.add_parser(
        "openapi", help="Update OpenAPI documentation"
    )
    openapi_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the OpenAPI documentation needs updating",
    )

    md_parser = subparsers.add_parser("md", help="Update launcher and supported models")
    md_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the launcher documentation needs updating",
    )

    args = parser.parse_args()

    if args.command == "openapi":
        openapi(args)
    elif args.command == "md":
        check_cli(args.check)
        check_supported_models(args.check)
