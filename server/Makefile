gen-server:
	# Compile protos
	pip install grpcio-tools==1.49.1 --no-cache-dir
	mkdir text_generation/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation/pb --grpc_python_out=text_generation/pb ../proto/generate.proto
	find text_generation/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation/pb/__init__.py

install-transformers:
	# Install specific version of transformers with custom cuda kernels
	rm transformers || true
	rm transformers-text_generation_inference || true
	curl -L -O https://github.com/OlivierDehaene/transformers/archive/refs/heads/text_generation_inference.zip
	unzip text_generation_inference.zip
	rm text_generation_inference.zip
	mv transformers-text_generation_inference transformers
	cd transformers && python setup.py install

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

install: gen-server install-torch install-transformers
	pip install pip --upgrade
	pip install -e . --no-cache-dir

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation/cli.py serve bigscience/bloom-560m --sharded