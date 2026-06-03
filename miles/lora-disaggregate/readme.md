# lora-disaggregate

```
docker create --gpus all --network host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-lora-disaggregate radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-lora-disaggregate
docker exec -it sglang-rl-yishenggong-lora-disaggregate bash

rm -r miles
git clone https://github.com/gongyisheng/miles.git
cd miles
git checkout miles-lora-disaggregate-mode-2

# install misc dep
# pip install flashinfer-jit-cache==0.6.7.post2 --index-url https://flashinfer.ai/whl/cu129
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
// hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct
```