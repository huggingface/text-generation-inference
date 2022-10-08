gen-server:
	mkdir bloom_inference/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=bloom_inference/pb --grpc_python_out=bloom_inference/pb ../proto/generate.proto
	find bloom_inference/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch bloom_inference/pb/__init__.py

unit-tests:
	python -m pytest --cov=bloom_inference tests

unit-tests-reporting:
	python -m pytest --junitxml=report.xml --cov=bloom_inference tests

pip-install:
	pip install grpcio-tools
	make gen-server
	pip install .

install:
	poetry install
	make gen-server