mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
root_dir := "${mkfile_dir}/../.."

.PHONY:	image run-local-dev-container install-dependencies install-server install-router install-launcher local-dev-install

image:
	docker build -t tgi-gaudi -f ${root_dir}/Dockerfile_gaudi ${root_dir}

run-local-dev-container:
		docker run -it \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
		-e LOG_LEVEL=debug \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=`cat /home/ubuntu/.cache/huggingface/token` \
		-e ENABLE_HPU_GRAPH=true \
		-e LIMIT_HPU_GRAPH=true \
		-e USE_FLASH_ATTENTION=true \
		-e FLASH_ATTENTION_RECOMPUTE=true \
		-e PORT=8080 \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /home/ubuntu/.cache/huggingface:/data \
		-v $(PWD):/text-generation-inference \
		-w /text-generation-inference \
		vault.habana.ai/gaudi-docker/1.19.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.1:latest

install-dependencies:
	pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
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
