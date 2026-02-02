# miles - lora megatron dev

## env
ssh radixark02
```
# use specific GPUs (e.g., GPU 0,1,2)
docker create --gpus '"device=0,1,2,3"' --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong radixark/miles:latest sleep infinity

docker start sglang-rl-yishenggong
docker exec -it sglang-rl-yishenggong bash

# install miles
rm -r miles
git clone --branch miles-lora-megatron --single-branch https://github.com/yushengsu-thu/miles.git 
cd miles
pip install -e .
cd ..

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python"
cd ..

# install megatron bridge
git clone --branch merged-megatron-0.16.0rc0 --single-branch https://github.com/yushengsu-thu/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..
```

## model and dataset
```
export MODEL_NAME=Qwen3-4B
mkdir -p /root/models /root/datasets
hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}
```