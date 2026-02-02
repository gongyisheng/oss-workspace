import miles.utils.external_utils.command_utils as U
import os

MODEL_NAME = "Qwen3-4B"
ENABLE_LORA = U.get_bool_env_var("ENABLE_LORA", "0")
ENABLE_LORA = True
ENABLE_EVAL = True

CUDA_VISIBLE_DEVICES = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    WANDB_API_KEY = "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"

    wandb_args = (
        "--use-wandb "
        "--wandb-project miles-lora-test "
        "--wandb-group qwen3-4B-megatron-lora-dapo-lr2e-5 "
        f"--wandb-key {WANDB_API_KEY} "
        "--disable-wandb-random-suffix "
    )

    lora_args = (
        (
            "--lora-rank 32 "
            "--lora-alpha 32 "
            "--lora-dropout 0.0 " # + from fsdp
            "--target-modules all-linear "
            f"--save /root/models/{MODEL_NAME}-lora-ckpt "
        )
        if ENABLE_LORA
        else ""
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 60} "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 128 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 5 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 2 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {'2e-5' if ENABLE_LORA else '1e-6'} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-decode-log-interval 1000 " "--sglang-enable-metrics "

    megatron_args = (
        "--no-offload-train "
        "--no-offload-rollout "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--colocate "
        "--offload-rollout-level kv_cache weight "
        "--train-backend fsdp "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{wandb_args} "
        f"{eval_args} "
        f"{megatron_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type=None,
    )


if __name__ == "__main__":
    prepare()
    execute()
