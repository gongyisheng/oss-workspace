# megatron bridge upgrade

```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-bridge-upgrade radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-bridge-upgrade
docker exec -it sglang-rl-yishenggong-bridge-upgrade bash

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout sglang-miles-v0.5.10
pip install -e "python"
cd ..

# install megatron bridge
git clone https://github.com/radixark/Megatron-Bridge.git
cd Megatron-Bridge
git checkout bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..

# install misc dep
pip install flashinfer-jit-cache==0.6.7.post2 --index-url https://flashinfer.ai/whl/cu129
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
```