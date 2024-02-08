include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm
include Makefile-awq
include Makefile-eetq
include Makefile-selective-scan

unit-tests:
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-megablocks:
	pip install git+https://github.com/OlivierDehaene/megablocks@181709df192de9a941fdf3a641cdc65a0462996e

install: gen-server
	pip install pip --upgrade
	pip install -r requirements_cuda.txt
	pip install -e ".[bnb, accelerate, quantize, peft]"

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

export-requirements:
	poetry export -o requirements_cuda.txt --extras bnb --without-hashes
	poetry export -o requirements_rocm.txt --without-hashes
