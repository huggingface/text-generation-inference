include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm
include Makefile-awq
include Makefile-eetq
include Makefile-selective-scan
include Makefile-lorax-punica
include Makefile-exllamav2
include Makefile-flashinfer

unit-tests:
	pip install -U pip uv
	uv venv
	uv pip install -e ".[dev]"
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install -U pip uv
	uv venv
	uv pip install ".[gen]"
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto/v3 --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/v3/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-server: gen-server
	uv pip install -e ".[accelerate, compressed-tensors, quantize, peft, outlines]"


install: install-cuda
	echo "Installed server"

install-cuda: install-server install-flash-attention-v2-cuda install-flash-attention
	uv pip install -e ".[attention,bnb,marlin,moe]"
	uv pip install nvidia-nccl-cu12==2.22.3

install-rocm: install-server install-flash-attention-v2-rocm  install-vllm-rocm
