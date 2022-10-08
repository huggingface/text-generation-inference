# BLOOM Inference

A Rust and gRPC server for BLOOM Inference.

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

- [ ] Improve model download
  - Store "shardable" layers separately and layer by layer
- [ ] Add batching args to router CLI 
- [ ] Add docstrings + comments everywhere as the codebase is fairly complicated
- [ ] Add tests
- [ ] Add shutdown logic in router and server
- [ ] Improve multi-processing logic in server
- [ ] Improve error handling everywhere
- [ ] Improve past key layer indexing?