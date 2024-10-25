include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm
include Makefile-awq
include Makefile-eetq
include Makefile-selective-scan
include Makefile-lorax-punica
include Makefile-fbgemm
include Makefile-exllamav2
include Makefile-flashinfer

unit-tests:
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 'types-protobuf' --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto/v3 --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/v3/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-server: gen-server
	pip install pip --upgrade
	pip install -r requirements_cuda.txt
	pip install -e ".[accelerate, quantize, peft, outlines]"


install: install-cuda
	echo "Installed server"

install-cuda: install-server install-flash-attention-v2-cuda install-vllm-cuda install-flash-attention install-fbgemm
	pip install -e ".[bnb,marlin,moe]"
	pip install nvidia-nccl-cu12==2.22.3

install-rocm: install-server install-flash-attention-v2-rocm  install-vllm-rocm

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

export-requirements:
	poetry export -o requirements_cuda.txt --without-hashes
	poetry export -o requirements_rocm.txt --without-hashes
	poetry export -o requirements_intel.txt --without-hashes
