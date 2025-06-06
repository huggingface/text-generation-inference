include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm
include Makefile-awq
include Makefile-selective-scan
include Makefile-exllamav2
include Makefile-flashinfer

unit-tests:
	pip install -U pip uv
	uv pip install -e ".[dev]"
	uv sync --inexact --extra dev --active
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install -U pip uv
	uv pip install -r requirements_gen.txt
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto/v3 --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/v3/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

gen-server-raw:
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto/v3 --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/v3/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-server: gen-server
	uv sync --inexact --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines --active


install: install-cuda
	echo "Installed server"

install-cuda: install-server install-flash-attention-v2-cuda install-flash-attention
	uv sync --inexact --extra attention --extra bnb --active
	uv pip install nvidia-nccl-cu12==2.22.3
	kernels download .

install-rocm: install-server install-flash-attention-v2-rocm  install-vllm-rocm

export-requirements:
	uv pip compile pyproject.toml --extra gen -o requirements_gen.txt --python-version 3.11
	uv pip compile pyproject.toml --extra bnb --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines -o requirements_cuda.txt --python-version 3.11
	uv pip compile pyproject.toml --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines -o requirements_intel.txt --python-version 3.11
	uv pip compile pyproject.toml --extra accelerate --extra compressed-tensors --extra quantize --extra peft --extra outlines -o requirements_rocm.txt --python-version 3.11
