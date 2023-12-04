## 通信优化说明文档

### 优化内容
    1、allreduce算子
    2、allgather_into_tensor算子

### 使用方法
    1、拉取镜像docker pull sakurahua/lm_inference:tgi-dev
    2、启动镜像docker run -it --rm --entrypoint /bin/bash --gpus all --net=host --shm-size=4G  -v xxx:/code sakurahua/lm_inference:tgi-dev
    3、启动服务
    USE_CUSTOM_NCCL=1 CUDA_VISIBLE_DEVICES=0,1 /root/.cargo/bin/text-generation-launcher --model-id /code/models/llama-7b-hf --port 7777 --sharded false
    4、验证服务
    curl localhost:7777/generate -X POST -d '{"inputs":"who are you?","parameters":{"max_new_tokens":100,"details":false}}' -H 'Content-Type: application/json'


### 注意点
    1、USE_CUSTOM_NCCL=1 开启通信算子，默认为0关闭
    2、USE_TP_EMBEDDING=0 关闭embedding并行，默认为1开启
    3、启用通信优化需搭配自定义启动命令/root/.cargo/bin/text-generation-launcher，如不启用优化，则使用原始的text-generation-launcher即可
    4、启用通信优化--sharded false该参数为必须，其他参数与tgi保持一致。

