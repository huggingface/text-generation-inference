gen-server:
	# Compile protos
	pip install grpcio-tools==1.49.1 --no-cache-dir
	mkdir bloom_inference/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=bloom_inference/pb --grpc_python_out=bloom_inference/pb ../proto/generate.proto
	find bloom_inference/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch bloom_inference/pb/__init__.py

install-transformers:
	# Install specific version of transformers
	rm transformers || true
	rm transformers-46d37bece7d3ffdef97b1ee4a3170c0a0627d921 || true
	curl -L -O https://github.com/huggingface/transformers/archive/46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	unzip 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	rm 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	mv transformers-46d37bece7d3ffdef97b1ee4a3170c0a0627d921 transformers
	cd transformers && python setup.py install

install-safetensors:
	# Install specific version of safetensors
	pip install setuptools_rust
	rm safetensors || true
	rm safetensors-634deccbcbad5eaf417935281f8b3be7ebca69c5 || true
	curl -L -O https://github.com/huggingface/safetensors/archive/634deccbcbad5eaf417935281f8b3be7ebca69c5.zip
	unzip 634deccbcbad5eaf417935281f8b3be7ebca69c5.zip
	rm 634deccbcbad5eaf417935281f8b3be7ebca69c5.zip
	mv safetensors-634deccbcbad5eaf417935281f8b3be7ebca69c5 safetensors
	cd safetensors/bindings/python && python setup.py develop

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

install: gen-server install-torch install-transformers install-safetensors
	pip install pip --upgrade
	pip install -e . --no-cache-dir

run-dev:
	python -m torch.distributed.run --nproc_per_node=2 bloom_inference/cli.py serve bigscience/bloom-560m --sharded