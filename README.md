# Text Generation Inference

A Rust and gRPC server for text generation inference.

## Load Tests

See `k6/load_test.js`
We send the default examples with a 1 second delay between each request.

Stages: 
- Ramp up to 50 concurrent requests per second in 1min
- Ramp up from 50 to 100 concurrent requests per second in 2min
- Ramp down to 0 concurrent requests per second in 1min


|                        | avg       | min       | med       | max        | p(90)     | p(95)     | RPS      |
|------------------------|-----------|-----------|-----------|------------|-----------|-----------|----------|
| Original code          | 8.9s      | 1s        | 9.12s     | 16.69s     | 13.7s     | 14.26s    | 5.9      |
| ISO with original code | 8.88s     | 959.53ms  | 8.89s     | 17.08s     | 13.34s    | 14.12s    | 5.94     |
| New batching logic     | **5.44s** | **1.27s** | **5.28s** | **13.12s** | **7.78s** | **8.92s** | **9.08** |

## Install

```shell
cd server
pip install .
```

```
cd router
cargo build --release
```

## Run

```shell
python server/bloom_inference/main.py bigscience/bloom --num-gpus 8 --shard-directory /dev/shm/models
```

```shell
./router/target/release/router
```

## TODO:

- [ ] Add docstrings + comments everywhere as the codebase is fairly complicated
- [ ] Add tests
- [ ] Add shutdown logic in router and server
- [ ] Improve multi-processing logic in server
- [ ] Improve past key layer indexing?