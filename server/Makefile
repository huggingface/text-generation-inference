gen-server:
	mkdir bloom_inference/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=bloom_inference/pb --grpc_python_out=bloom_inference/pb ../proto/generate.proto
	find bloom_inference/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch bloom_inference/pb/__init__.py

install-transformers:
	# Install specific version of transformers
	rm transformers || true
	wget https://github.com/huggingface/transformers/archive/46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	unzip 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	rm 46d37bece7d3ffdef97b1ee4a3170c0a0627d921.zip
	mv transformers-46d37bece7d3ffdef97b1ee4a3170c0a0627d921 transformers
	cd transformers && python setup.py install

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

pip-install:
	pip install grpcio-tools
	make gen-server
	make install-torch
	make install-transformers
	pip install .

install:
	poetry install
	make gen-server
	make install-torch
	make install-transformers
