# lora weight sync
env
```
# sgl dev
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong radixark/miles:dev sleep infinity

# home
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng/Documents/checkpoints:/data --name miles-dev radixark/miles sleep infinity
docker start miles-dev
docker exec -it miles-dev bash
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B-Instruct
```