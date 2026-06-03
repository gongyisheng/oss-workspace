ENV="HEAD_NODE_IP=10.0.0.243 \
    TRAIN_GPUS_PER_NODE=1 \
    NUM_TRAIN_GPUS=1 \
    NUM_ROLLOUT_GPUS=2 \
    ROLLOUT_GPUS_PER_NODE=2 \
    ROLLOUT_GPUS_PER_ENGINE=2"

env ${ENV} bash examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh broadcast 0
env ${ENV} bash examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh broadcast 1