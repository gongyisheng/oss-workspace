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
```