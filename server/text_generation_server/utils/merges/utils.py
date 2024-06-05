# coding=utf-8
# From: https://github.com/huggingface/peft/pull/1364
# Copyright 2024-present the HuggingFace Inc. team.
# Modifications by Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

import torch


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.reshape(-1).shape[0])
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: torch.Tensor,
    density: float,
    method: Literal["magnitude", "random"],
    rescale: bool = False,
) -> torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
    rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.
    """
    if density >= 1:
        return tensor
    elif density < 0:
        raise ValueError("Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
):
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
    tensor (`torch.Tensor`):The tensor to get the mask from.
    method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = (sign * tensor.abs()).sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors, majority_sign_mask):
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)
