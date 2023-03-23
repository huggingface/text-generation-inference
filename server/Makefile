transformers_commit := 2b57aa18da658e7d2f42ef6bd5b56751af582fef
flash_att_commit := 4d87e4d875077ad9efd25030efa4ab0ba92c19e1

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
	git clone https://github.com/OlivierDehaene/transformers.git
	cd transformers && git checkout $(transformers_commit)
	cd transformers && python setup.py install

install-flash-attention:
	# Install specific version of flash attention
	pip install packaging
	pip uninstall flash_attn rotary_emb dropout_layer_norm -y || true
	rm -rf flash-attention || true
	git clone https://github.com/HazyResearch/flash-attention.git
	cd flash-attention && git checkout $(flash_att_commit)
	cd flash-attention && python setup.py install && cd csrc/layer_norm && python setup.py install && cd ../rotary && python setup.py install

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

install: gen-server install-torch install-transformers
	pip install pip --upgrade
	pip install -e . --no-cache-dir

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded