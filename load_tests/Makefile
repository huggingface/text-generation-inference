.PHONY: download-dataset load-test build-k6
download-dataset:
	@if [ ! -f ./benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json ]; then \
		echo "Downloading dataset"; \
		curl -L -o benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json; \
	else \
		echo "Dataset already downloaded"; \
	fi

load-test: download-dataset build-k6
	poetry install && poetry run python load_test.py

build-k6:
	mkdir -p /tmp/xk6 && \
	cd /tmp/xk6 && \
	git clone https://github.com/mstoykov/xk6-sse.git && \
	cd xk6-sse && \
	git checkout useSobek && \
	go install go.k6.io/xk6/cmd/xk6@latest && \
	xk6 build --with github.com/phymbert/xk6-sse=. && \
	mkdir -p ~/.local/bin/ && \
	mv k6 /tmp/k6-sse && \
	rm -rf /tmp/xk6 && \
	/tmp/k6-sse --version