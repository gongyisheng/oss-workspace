# lora support for gpt-oss
env
```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-gpt-oss-old radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-gpt-oss-old
docker exec -it sglang-rl-yishenggong-gpt-oss-old bash

# install miles
rm -r miles
git clone --branch miles-gpt-oss-moe-lora --single-branch https://github.com/gongyisheng/miles.git
cd miles
pip install -e .
cd ..

# install sglang
git clone --branch sglang-gpt-oss-moe-lora --single-branch https://github.com/gongyisheng/sglang.git
cd sglang
pip install -e "python"
cd ..

# install megatron bridge
git clone --branch merged-megatron-0.16.0rc0-gpt-oss-moe-lora --single-branch https://github.com/gongyisheng/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..

# install misc dep
pip install transformers==4.57.1
pip install flashinfer-jit-cache==0.6.6 --index-url https://flashinfer.ai/whl/cu129

# dep issues
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
megatron-bridge 0.3.0rc0 requires causal-conv1d, which is not installed.
megatron-bridge 0.3.0rc0 requires hydra-core<=1.3.2,>1.3, which is not installed.
megatron-bridge 0.3.0rc0 requires mamba-ssm, which is not installed.
megatron-bridge 0.3.0rc0 requires nvidia-resiliency-ext, which is not installed.
megatron-bridge 0.3.0rc0 requires open-clip-torch>=3.2.0, which is not installed.
megatron-bridge 0.3.0rc0 requires pyyaml>=6.0.2, but you have pyyaml 6.0.1 which is incompatible.
megatron-bridge 0.3.0rc0 requires transformer-engine[pytorch]<2.10.0,>=2.9.0a0, but you have transformer-engine 2.10.0 which is incompatible.
megatron-bridge 0.3.0rc0 requires transformers<5.0.0,>=4.57.1, but you have transformers 5.3.0 which is incompatible.
```

env new:
```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-gpt-oss-2 radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-gpt-oss-2
docker exec -it sglang-rl-yishenggong-gpt-oss-2 bash

# install miles
rm -r miles
git clone --branch miles-gpt-oss-moe-lora-yusheng --single-branch https://github.com/yushengsu-thu/miles.git
cd miles
pip install -e .
cd ..

# install sglang
git clone --branch sglang-miles-lora --single-branch https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python"
cd ..

# install megatron bridge
git clone --branch bridge --single-branch https://github.com/radixark/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
hf download openai/gpt-oss-20b --local-dir /root/gpt-oss-20b
```