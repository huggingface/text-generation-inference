# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Union


from transformers.image_utils import ImageInput, is_valid_image, is_pil_image


def is_valid_list_of_images(images: List):
    return images and all(is_valid_image(image) for image in images)


def make_nested_list_of_images(
    images: Union[List[ImageInput], ImageInput],
) -> ImageInput:
    """
    Ensure that the output is a nested list of images.
    Args:
        images (`Union[List[ImageInput], ImageInput]`):
            The input image.
    Returns:
        list: A list of list of images or a list of 4d array of images.
    """
    # If it's a list of batches, it's already in the right format
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) for images_i in images)
    ):
        return images

    # If it's a list of images, it's a single batch, so convert it to a list of lists
    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == 3:
            return [images]
        if images[0].ndim == 4:
            return [list(image) for image in images]

    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == 3:
            return [[images]]
        if images.ndim == 4:
            return [list(images)]

    raise ValueError(
        "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
    )
