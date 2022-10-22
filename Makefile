install-server:
	cd server && make install

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

install: install-server install-router install-launcher

server-dev:
	cd server && make run-dev

router-dev:
	cd router && cargo run

run-bloom-560m:
	text-generation-launcher --model-name bigscience/bloom-560m --num-shard 2

run-bloom:
	text-generation-launcher --model-name bigscience/bloom --num-shard 8
