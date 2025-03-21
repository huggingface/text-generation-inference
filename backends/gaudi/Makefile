mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
root_dir := "${mkfile_dir}/../.."

HABANA_VERSION := 1.20.0
PYTORCH_VERSION := 2.6.0

.PHONY:	image run-local-dev-container install-dependencies install-server install-router install-launcher local-dev-install

image:
	docker build -t tgi-gaudi -f ${root_dir}/Dockerfile_gaudi ${root_dir} --build-arg HABANA_VERSION=$(HABANA_VERSION) --build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) --build-arg http_proxy=${http_proxy}     --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy}

run-local-dev-container:
		docker run -it \
		--runtime=habana \
		--ipc=host \
		--cap-add=sys_nice \
		--net=host \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
		-e HF_TOKEN=`cat /home/ubuntu/.cache/huggingface/token` \
		-e LOG_LEVEL=debug \
		-e PORT=8080 \
		-v /home/ubuntu/.cache/huggingface:/data \
		-v $(PWD):/text-generation-inference \
		-w /text-generation-inference \
		vault.habana.ai/gaudi-docker/$(HABANA_VERSION)/ubuntu22.04/habanalabs/pytorch-installer-$(PYTORCH_VERSION):latest

install-dependencies:
	pip install git+https://github.com/HabanaAI/DeepSpeed.git@$(HABANA_VERSION)
	pip install outlines~=0.0.34
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

install-server:
	make -C ${root_dir}/backends/gaudi/server install PROTO_PATH=../../../proto/v3

install-router:
	make -C ${root_dir} install-router

install-launcher:
	make -C ${root_dir} install-launcher

# use source to load the rust in path
local-dev-install: install-dependencies
	bash -c 'source "$$HOME/.cargo/env" && \
		make install-server && \
		make install-router && \
		make install-launcher'
