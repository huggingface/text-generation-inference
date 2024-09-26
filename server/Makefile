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
	pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 'types-protobuf' --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto/v3 --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/v3/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install: gen-server
	pip install pip --upgrade
	pip install -r requirements.txt
	pip install -e "."

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

install-poetry:
	curl -sSL https://install.python-poetry.org | python3 -

update-lock:
	rm poetry.lock
	poetry lock --no-update

export-requirements:
	poetry export -o requirements.txt --without-hashes
