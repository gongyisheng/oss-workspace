# lora-disaggregate

```
docker create --gpus all --network host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-lora-disaggregate radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-lora-disaggregate
docker exec -it sglang-rl-yishenggong-lora-disaggregate bash

git clone https://github.com/gongyisheng/sglang.git
cd sglang
git checkout sglang-miles-lora-disaggregate-mode-2
pip install -e "python"
cd ..

# install misc dep
pip list | grep flashinfer
pip install flashinfer-jit-cache==0.6.11.post1 --index-url https://flashinfer.ai/whl/cu129
# pip install --no-deps --force-reinstall "torchvision==0.26.0+cu130" --index-url https://download.pytorch.org/whl/cu130
# echo "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib" > /etc/ld.so.conf.d/nvidia-cu13.conf && ldconfig

rm -r miles
git clone https://github.com/gongyisheng/miles.git
cd miles
git checkout miles-lora-disaggregate-mode-2
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
// hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct
```