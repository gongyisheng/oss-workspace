#!/bin/bash
# Start SGLang server with Qwen2.5-3B-Instruct for weight update reproduction.
# Config mirrors _compute_server_args() in sglang_engine.py.
#
# Usage: bash scripts/repro/start_sglang_server.sh [MODEL_PATH] [PORT]

MODEL_PATH="${1:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${2:-30000}"
HOST="127.0.0.1"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
LORA_RANK="${LORA_RANK:-32}"

echo "Starting SGLang server..."
echo "  Model:   ${MODEL_PATH}"
echo "  Host:    ${HOST}:${PORT}"
echo "  TP/DP:   ${TP_SIZE}/${DP_SIZE}"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --skip-server-warmup \
    --enable-memory-saver \
    --enable-draft-weights-cpu-backup \
    --enable-lora \
    --max-loras-per-batch 1 \
    --max-lora-rank "${LORA_RANK}" \
    --lora-target-modules all-linear
