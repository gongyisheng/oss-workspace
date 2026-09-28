#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=1
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# for rerun the task
pkill sglang
ray stop --force
sleep 5 # Wait for processes to terminate gracefully
# Force kill any remaining processes.
# Note: `pkill -9 python` is broad and can be risky.
pkill -9 sglang
pkill -9 ray
pkill -9 python

set -ex


MILES_ROOT=/root/miles
MODEL_ARGS_LINE="$(python3 "${MILES_ROOT}/miles/utils/external_utils/model_args_utils.py" "qwen2.5-3B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen2.5-3B-Instruct/
   --megatron-to-hf-mode bridge
   --save /root/checkpoints/lora/
   --save-interval 10
)

LORA_ARGS=(
   --lora-rank 32                    # LoRA rank (typical values: 8, 16, 32, 64)
   --lora-alpha 32                   # LoRA alpha (usually 2x rank)
   --lora-dropout 0.0                # LoRA dropout (0.0 for RL training)
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --over-sampling-batch-size 32
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # not compatable with fused attn backend
   # --use-dynamic-batch-size
   # --max-tokens-per-gpu 8192
   --qkv-format bshd
   --micro-batch-size 8
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss # if use kl loss, should use --ref-load
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5 # Higher LR often works better for LoRA
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project miles-rl-dev
   --wandb-group qwen2.5-3B-dapo-lora
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # --sglang-mem-fraction-static 0.7
   --sglang-mem-fraction-static 0.4

   # --sglang-enable-deterministic-inference
   # --sglang-attention-backend flashinfer
   # --deterministic-mode
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # --attention-backend flash # flash attn, does not work on sm120
   --attention-backend fused
)


# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   --colocate \
   --calculate-per-token-loss \
   --use-miles-router \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${ROLLOUT_ARGS[@]}
