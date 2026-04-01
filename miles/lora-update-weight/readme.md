# lora weight sync
env
```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-2 radixark/miles:dev sleep infinity
dokcer start sglang-rl-yishenggong-2
docker exec -it sglang-rl-yishenggong-2 bash
```

## model and dataset
```
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/qwen3-4b
```