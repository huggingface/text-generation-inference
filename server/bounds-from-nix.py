#!/usr/bin/env python3

import json
import subprocess
from typing import Dict, Union
import toml

# Special cases that have download URLs.
SKIP = {"attention-kernels", "marlin-kernels", "moe-kernels"}


def is_optional(info: Union[str, Dict[str, str]]) -> bool:
    return isinstance(info, dict) and "optional" in info and info["optional"]


if __name__ == "__main__":
    with open("pyproject.toml") as f:
        pyproject = toml.load(f)

    nix_packages = json.loads(
        subprocess.run(
            ["nix", "develop", ".#server", "--command", "pip", "list", "--format=json"],
            stdout=subprocess.PIPE,
        ).stdout
    )

    nix_packages = {pkg["name"]: pkg["version"] for pkg in nix_packages}

    packages = []
    optional_packages = []

    for package, info in pyproject["tool"]["poetry"]["dependencies"].items():
        if package in nix_packages and package not in SKIP:
            if is_optional(info):
                optional_packages.append(f'"{package}@^{nix_packages[package]}"')
            else:
                packages.append(f'"{package}@^{nix_packages[package]}"')

    print(f"poetry add {' '.join(packages)}")
    print(f"poetry add --optional {' '.join(optional_packages)}")
