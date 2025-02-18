#  Copyright 2025 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
root_dir := "${mkfile_dir}/../.."

.PHONY:	image install_server test_server test_integration

VERSION := $(shell gawk 'match($$0, /^version = "(.*)"/, a) {print a[1]}' ${root_dir}/Cargo.toml)

image:
	docker build --rm -f ${root_dir}/Dockerfile.neuron \
				 --ulimit nofile=100000:100000 \
				 --build-arg VERSION=$(VERSION) \
				 -t text-generation-inference:$(VERSION)-neuron ${root_dir}
	docker tag text-generation-inference:$(VERSION)-neuron text-generation-inference:latest-neuron

install_server:
	make -C ${mkfile_dir}/server install VERSION:=${VERSION}

test_server: install_server
	python -m pip install -r ${mkfile_dir}/tests/requirements.txt
	python -m pytest -sv ${mkfile_dir}/tests/server
