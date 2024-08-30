.PHONY: download-dataset load-test build-k6
download-dataset:
	@if [ ! -f ./benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json ]; then \
		echo "Downloading dataset"; \
		curl -L -o benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json; \
	else \
		echo "Dataset already downloaded"; \
	fi

load-test: export PATH := $(shell pwd)/.bin/:$(PATH)
load-test: download-dataset
	poetry install && poetry run python load_test.py

build-k6:
	go install go.k6.io/xk6/cmd/xk6@latest && \
	xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182 && \
	mkdir -p .bin/ && \
	mv k6 .bin/k6 && \
	.bin/k6 --version