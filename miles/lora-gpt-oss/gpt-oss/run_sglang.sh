#!/bin/bash
set -ex

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

export CUDA_VISIBLE_DEVICES=7

MODEL_PATH=${MODEL_PATH:-/root/gpt-oss-20b}
PORT=${PORT:-30000}


python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --lora-paths "my-lora-adapter=/root/lora_model/gpt-oss-20b" \
    --tp-size 1 \
    --lora-backend triton \
    --port "${PORT}" \
    --host 0.0.0.0
