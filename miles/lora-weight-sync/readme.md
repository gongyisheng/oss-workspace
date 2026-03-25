# lora weight sync
env
```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-2 radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-2
docker exec -it sglang-rl-yishenggong-2 bash

# install miles
cd miles
pip install -e .
cd ..

# install megatron bridge
git clone --branch merged-megatron-0.16.0rc0 --single-branch https://github.com/yushengsu-thu/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..

# install misc dep
pip install flashinfer-jit-cache==0.6.4 --index-url https://flashinfer.ai/whl/cu129
```

## model and dataset
```
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/qwen3-4b
```