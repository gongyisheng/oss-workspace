# multi-lora training

Train multiple LoRA adapters (all-linear, excluding experts) per step.
miles [PR #1141](https://github.com/radixark/miles/pull/1141), megatron-bridge [PR #4](https://github.com/radixark/Megatron-Bridge/pull/4).

env
```
docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-multi-lora radixark/miles:dev sleep infinity
docker start sglang-rl-yishenggong-multi-lora
docker exec -it sglang-rl-yishenggong-multi-lora bash

# install miles
rm -r miles
git clone --branch feat/multilora-rebase --single-branch https://github.com/mathewjhan/miles.git
cd miles
pip install -e .
cd ..

# install megatron bridge
git clone --branch radixark/multilora-support --single-branch https://github.com/mathewjhan/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..

# sglang: stock build in radixark/miles:dev is used; multi-lora selects the
# triton backend at runtime via --sglang-lora-backend triton (no reinstall needed)
```

## model and dataset

`provision.sh` downloads Qwen3-4B + datasets and converts the HF checkpoint to torch_dist:
```
cd miles
bash examples/multi_lora/provision.sh
```

equivalent manual downloads:
```
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
```

## run

normal training (stops when all adapters are done):
```
# configure W&B credentials in examples/multi_lora/single_run.sh first
bash examples/multi_lora/single_run.sh |& tee run.log
```

as a long-running service (online load/unload of adapters):
```
# configure W&B credentials in examples/multi_lora/start_service.sh first
bash examples/multi_lora/start_service.sh |& tee run.log   # shell 1
bash examples/multi_lora/submit_schedule.sh               # shell 2
```

checkpoints and LoRA safetensors are saved under `examples/multi_lora/adapters/*/checkpoints`.
