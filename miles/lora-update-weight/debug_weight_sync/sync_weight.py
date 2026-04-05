"""
Reproduce garbage output by sending base weights + LoRA adapter to SGLang.

Flow (mirrors UpdateWeightFromTensor):
  1. Load model weights from HF checkpoint
  2. Serialize via FlattenedTensorBucket + MultiprocessingSerializer
  3. POST /update_weights_from_tensor (base weights)
  4. POST /load_lora_adapter_from_tensors (LoRA adapter)
  5. Run inference to observe output quality

Usage:
  # First start server: bash scripts/repro/start_sglang_server.sh
  python scripts/repro/repro_garbage_output.py --model-path Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket


def serialize_named_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> list[str]:
    """Serialize tensors the same way miles does in _send_to_colocated_engine."""
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        groups = {"mixed": named_tensors}
    else:
        groups = {}
        for name, tensor in named_tensors:
            dt = tensor.dtype
            if dt not in groups:
                groups[dt] = []
            groups[dt].append((name, tensor))

    serialized = []
    for _dtype, tensors in groups.items():
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized.append(MultiprocessingSerializer.serialize(data, output_str=True))
    return serialized


def generate(base_url: str, prompt: str, max_tokens: int = 64, lora_name: str | None = None) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
    }
    if lora_name:
        payload["lora_path"] = lora_name
    resp = requests.post(f"{base_url}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["text"]


def send_base_weights(base_url: str, named_tensors: list[tuple[str, torch.Tensor]], version: str = "1"):
    """POST /update_weights_from_tensor — same as miles _send_base_params."""
    serialized = serialize_named_tensors(named_tensors)
    # In real miles, serialized_named_tensors is a list (one per TP rank).
    # For single-GPU, we just wrap in a list.
    payload = {
        "serialized_named_tensors": serialized,
        "load_format": "flattened_bucket",
        "flush_cache": False,
        "weight_version": version,
    }
    resp = requests.post(f"{base_url}/update_weights_from_tensor", json=payload)
    resp.raise_for_status()
    result = resp.json()
    print(f"  Base weight update result: {result}")
    return result


def send_lora_weights(
    base_url: str,
    named_tensors: list[tuple[str, torch.Tensor]],
    lora_config: dict,
    lora_name: str = "miles_lora",
    unload_first: bool = False,
):
    """POST /load_lora_adapter_from_tensors — same as miles _send_lora_params."""
    if unload_first:
        print(f"  Unloading existing adapter '{lora_name}'...")
        resp = requests.post(f"{base_url}/unload_lora_adapter", json={"lora_name": lora_name})
        resp.raise_for_status()

    serialized = serialize_named_tensors(named_tensors)
    payload = {
        "lora_name": lora_name,
        "serialized_tensors": serialized[0],  # LoRA uses only first dtype group
        "config_dict": lora_config,
        "load_format": "flattened_bucket",
        "pinned": False,
    }
    resp = requests.post(f"{base_url}/load_lora_adapter_from_tensors", json=payload)
    resp.raise_for_status()
    result = resp.json()
    print(f"  LoRA weight load result: {result}")
    return result


def build_lora_config(rank: int = 8, alpha: int = 16) -> dict:
    """Same as miles build_lora_sync_config."""
    return {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def create_random_lora_weights(
    model: AutoModelForCausalLM,
    target_modules: list[str],
    rank: int = 8,
) -> list[tuple[str, torch.Tensor]]:
    """Create random LoRA A/B matrices for all target modules in the model.

    TODO(user): Replace with real trained LoRA weights if you have them.
    Random weights are intentionally used here to reproduce garbage output.
    """
    lora_tensors = []
    for name, param in model.named_parameters():
        # Match target module names (e.g. "model.layers.0.self_attn.q_proj.weight")
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        short_name = module_name.rsplit(".", 1)[-1]
        if short_name not in target_modules or not name.endswith(".weight"):
            continue

        out_features, in_features = param.shape
        # LoRA weight naming convention for SGLang/PEFT
        base_key = name.replace(".weight", "")
        lora_a = torch.randn(rank, in_features, dtype=param.dtype, device="cpu") * 0.01
        lora_b = torch.zeros(out_features, rank, dtype=param.dtype, device="cpu")
        lora_tensors.append((f"base_model.model.{base_key}.lora_A.default.weight", lora_a))
        lora_tensors.append((f"base_model.model.{base_key}.lora_B.default.weight", lora_b))

    return lora_tensors


def main():
    parser = argparse.ArgumentParser(description="Reproduce garbage output with base + LoRA weight sync")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="What is 2+2? Answer concisely.")
    parser.add_argument("--skip-base-sync", action="store_true", help="Skip base weight sync, only send LoRA")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    target_modules = "all-linear"
    lora_config = build_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)

    # --- Step 0: Baseline inference ---
    print("\n=== Step 0: Baseline inference (original server weights) ===")
    output = generate(args.base_url, args.prompt)
    print(f"  Prompt: {args.prompt}")
    print(f"  Output: {output}")

    # --- Step 1: Send base weights ---
    if not args.skip_base_sync:
        print("\n=== Step 1: Sending base weights ===")
        base_tensors = [(name, param.data.cpu()) for name, param in model.named_parameters()]
        print(f"  Total base params: {len(base_tensors)}")
        # Send in chunks to avoid OOM (same as miles chunked iteration)
        chunk_size = 50
        for i in range(0, len(base_tensors), chunk_size):
            chunk = base_tensors[i : i + chunk_size]
            print(f"  Sending chunk {i // chunk_size + 1} ({len(chunk)} params)...")
            send_base_weights(args.base_url, chunk, version="1")

        print("\n=== Step 1b: Inference after base weight sync ===")
        output = generate(args.base_url, args.prompt)
        print(f"  Output: {output}")

    # --- Step 2: Send LoRA adapter ---
    print("\n=== Step 2: Sending LoRA adapter weights ===")
    lora_tensors = create_random_lora_weights(model, target_modules, rank=args.lora_rank)
    print(f"  Total LoRA params: {len(lora_tensors)}")
    send_lora_weights(args.base_url, lora_tensors, lora_config, lora_name="miles_lora")

    # --- Step 3: Inference with LoRA ---
    print("\n=== Step 3: Inference with LoRA adapter ===")
    output = generate(args.base_url, args.prompt, lora_name="miles_lora")
    print(f"  Output (with LoRA): {output}")

    # --- Step 4: Inference without LoRA (base only) ---
    print("\n=== Step 4: Inference WITHOUT LoRA (base only after sync) ===")
    output = generate(args.base_url, args.prompt)
    print(f"  Output (base only): {output}")

    print("\n=== Done. Compare outputs above for garbage/degradation. ===")

    del model


if __name__ == "__main__":
    main()
