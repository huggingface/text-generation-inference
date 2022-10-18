install-server:
	cd server && make pip-install

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

install:
	make install-server
	make install-router
	make install-launcher

run-bloom-560m:
	text-generation-launcher --model-name bigscience/bloom-560m --shard-directory /tmp/models --num-shard 2

run-bloom:
	text-generation-launcher --model-name bigscience/bloom --shard-directory /tmp/models --num-shard 8
