#!/bin/bash
# Start SGLang server with Qwen2.5-3B-Instruct for weight update reproduction.
# Config mirrors _compute_server_args() in sglang_engine.py.
#
# Usage: bash scripts/repro/start_sglang_server.sh [MODEL_PATH] [PORT]

pkill sglang
ray stop --force
sleep 5 # Wait for processes to terminate gracefully
pkill -9 sglang
pkill -9 ray
pkill -9 python

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

# Env vars matching miles rollout.py Ray runtime_env
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}"
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-true}"
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-true}"
export SGLANG_MEMORY_SAVER_CUDA_GRAPH="${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-true}"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT="${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT:-true}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-false}"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE:-false}"

MODEL_PATH="${1:-/root/Qwen2.5-3B-Instruct}"
PORT="${2:-30000}"
HOST="127.0.0.1"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
LORA_RANK="${LORA_RANK:-32}"
RANDOM_SEED="${RANDOM_SEED:-42}"

echo "Starting SGLang server..."
echo "  Model:   ${MODEL_PATH}"
echo "  Host:    ${HOST}:${PORT}"
echo "  TP/DP:   ${TP_SIZE}/${DP_SIZE}"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --random-seed "${RANDOM_SEED}" \
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
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
