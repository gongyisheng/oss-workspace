# lora for GLM-5.1 / GLM-5.2 (MoE + MLA + DSA)

GRPO LoRA training via the Megatron-Bridge path (`--megatron-to-hf-mode bridge`).

## env
```
docker create --gpus all --network host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit nofile=65536:65536 --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v /home/yisheng:/workspace -v /data/cache/huggingface:/root/.cache/huggingface -v /data:/data --name sglang-rl-yishenggong-glm5 radixark/miles:bump-v0.5.13-test sleep infinity
docker start sglang-rl-yishenggong-glm5
docker exec -it sglang-rl-yishenggong-glm5 bash

# install sglang (PR sgl-project/sglang#28703 — DSA indexer LoRA targets)
git clone --branch sglang-miles-glm-dev --single-branch https://github.com/yushengsu-thu/sglang.git
cd sglang
pip install -e "python"
cd ..

# install megatron bridge (PR radixark/Megatron-Bridge#13 — glm_moe_dsa bridge build)
git clone --branch bridge-dev-glm --single-branch https://github.com/radixark/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
cd ..

# install miles (PR radixark/miles#1373 — launcher, registries, LoRA enablement)
rm -r miles
git clone --branch miles-dev-2026-06-16 --single-branch https://github.com/yushengsu-thu/miles.git
cd miles
pip install -e .
cd ..
```

## model and dataset
```
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

## run
```
python scripts/run_glm5_lora.py train --model-name GLM-5.1-6layer
python scripts/run_glm5_lora.py train --model-name GLM-5.2-7layer
```

## notes
- Requires `--qkv-format bshd` + `--micro-batch-size 1`: megatron-core's DSA core-attention needs a 4D query; the default `thd` packing yields a 3D query.
- GLM-5.2 adds DSA cross-layer index sharing (handled in Megatron-Bridge#13). sglang cannot yet serve the 5.2 cross-layer rollout, so GLM-5.2 is train-only; GLM-5.1 runs the full rollout→train loop.
- launcher: `scripts/run_glm5_lora.py`; model registries under `scripts/models/` (`glm5-744B-A40B_6layer.sh`, `glm5.2-744B-A40B_7layer.sh`).
