install-server:
	cd server && make install

install-integration-tests:
	cd integration-tests && pip install -r requirements.txt
	cd clients/python && pip install .

install-router:
	cd router && cargo install --locked --path .

install-launcher:
	cd launcher && cargo install --locked --path .

install-benchmark:
	cd benchmark && cargo install --locked --path .

install: install-server install-router install-launcher install-custom-kernels

server-dev:
	cd server && make run-dev

router-dev:
	cd router && cargo run -- --port 8080

rust-tests: install-router install-launcher
	cargo test

integration-tests: install-integration-tests
	pytest -s -vv -m "not private" integration-tests

update-integration-tests: install-integration-tests
	pytest -s -vv --snapshot-update integration-tests

python-server-tests:
	HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv -m "not private" server/tests

python-client-tests:
	pytest clients/python/tests

python-tests: python-server-tests python-client-tests

run-falcon-7b-instruct:
	text-generation-launcher --model-id tiiuae/falcon-7b-instruct --port 8080

clean:
	rm -rf target aml

debug_image_build:
	docker build --no-cache --progress=plain -t debug_tgi .
