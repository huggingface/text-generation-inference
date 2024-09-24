# Copyright 2023 The HuggingFace Team. All rights reserved.
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

__version__ = "0.7.0"

DEPRECATION_WARNING = (
    "`text_generation` clients are deprecated and will be removed in the near future. "
    "Please use the `InferenceClient` from the `huggingface_hub` package instead."
)

from text_generation.client import Client, AsyncClient
from text_generation.inference_api import InferenceAPIClient, InferenceAPIAsyncClient
