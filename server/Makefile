transformers_commit := 2b57aa18da658e7d2f42ef6bd5b56751af582fef

gen-server:
	# Compile protos
	pip install grpcio-tools==1.51.1 --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb --grpc_python_out=text_generation_server/pb ../proto/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-transformers:
	# Install specific version of transformers with custom cuda kernels
	pip uninstall transformers -y || true
	rm -rf transformers || true
	rm -rf transformers-$(transformers_commit) || true
	curl -L -O https://github.com/OlivierDehaene/transformers/archive/$(transformers_commit).zip
	unzip $(transformers_commit).zip
	rm $(transformers_commit).zip
	mv transformers-$(transformers_commit) transformers
	cd transformers && python setup.py install

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

install: gen-server install-torch install-transformers
	pip install pip --upgrade
	pip install -e . --no-cache-dir

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded