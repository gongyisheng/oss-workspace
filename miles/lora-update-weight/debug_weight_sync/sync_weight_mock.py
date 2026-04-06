"""
Reproduce garbage output by sending base weights + LoRA adapter to SGLang.

Flow (mirrors UpdateWeightFromTensor):
  1. Load model weights from HF checkpoint
  2. Serialize via FlattenedTensorBucket + MultiprocessingSerializer
  3. POST /update_weights_from_tensor (base weights)
  4. POST /load_lora_adapter_from_tensors (LoRA adapter)
  5. Run inference to observe output quality

Usage:
  # First start server: bash start_sglang.sh
  python sync_weight.py
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


def serialize_named_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[list[str], list]:
    """Serialize tensors the same way miles does in _send_to_colocated_engine.

    Returns (serialized_strings, cuda_refs). Caller MUST hold cuda_refs alive
    until the server has finished deserializing (i.e., until the HTTP response
    returns), because CUDA IPC handles reference the original GPU memory.
    """
    # Move tensors to CUDA. MultiprocessingSerializer uses ForkingPickler which
    # serializes GPU tensors via CUDA IPC (works cross-process without auth) but
    # serializes CPU tensors via FD-based resource_sharer (requires matching
    # authkeys — fails when client and server are independent processes).
    named_tensors = [(name, t.cuda()) for name, t in named_tensors]

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
    cuda_refs = []
    for _dtype, tensors in groups.items():
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        cuda_refs.append(data)
        serialized.append(MultiprocessingSerializer.serialize(data, output_str=True))
    return serialized, cuda_refs


def generate(base_url: str, prompt: str, max_tokens: int = 512, lora_name: str | None = None) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 1},
    }
    if lora_name:
        payload["lora_path"] = lora_name
    resp = requests.post(f"{base_url}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["text"]


# ---------------------------------------------------------------------------
# sglang memory-saver lifecycle helpers (mirrors miles colocate flow)
# ---------------------------------------------------------------------------

def release_memory_occupation(base_url: str, tags: list[str] | None = None):
    """Offload sglang GPU memory to CPU (flush_cache + release)."""
    resp = requests.get(f"{base_url}/flush_cache")
    resp.raise_for_status()
    resp = requests.post(f"{base_url}/release_memory_occupation", json={"tags": tags})
    resp.raise_for_status()
    print(f"  release_memory_occupation(tags={tags}): {resp.json()}")


def resume_memory_occupation(base_url: str, tags: list[str] | None = None):
    """Reload sglang GPU memory from CPU backup."""
    resp = requests.post(f"{base_url}/resume_memory_occupation", json={"tags": tags})
    resp.raise_for_status()
    print(f"  resume_memory_occupation(tags={tags}): {resp.json()}")


def pause_generation(base_url: str):
    resp = requests.post(f"{base_url}/pause_generation", json={})
    resp.raise_for_status()
    print("  pause_generation: ok")


def continue_generation(base_url: str):
    resp = requests.post(f"{base_url}/continue_generation", json={})
    resp.raise_for_status()
    print("  continue_generation: ok")


def flush_cache(base_url: str):
    resp = requests.get(f"{base_url}/flush_cache")
    resp.raise_for_status()
    print("  flush_cache: ok")


def try_inference(base_url: str, prompt: str, label: str, lora_name: str | None = None):
    """Try inference and print result. Returns True if request succeeded."""
    try:
        output = generate(base_url, prompt, lora_name=lora_name)
        tag = f"(lora={lora_name})" if lora_name else "(base)"
        print(f"[{label}] {tag} → {output[:512]}")
        return True
    except Exception as e:
        print(f"[{label}] FAILED: {e}")
        return False


def send_base_weights(base_url: str, named_tensors: list[tuple[str, torch.Tensor]], version: str = "1"):
    """POST /update_weights_from_tensor — same as miles _send_base_params."""
    serialized, cuda_refs = serialize_named_tensors(named_tensors)
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
    del cuda_refs  # Safe to free after server responded
    print(f"  Base weight update result: {result}")
    return result


def send_lora_weights(
    base_url: str,
    named_tensors: list[tuple[str, torch.Tensor]],
    lora_config: dict,
    lora_name: str = "miles_lora",
    unload_first: bool = True,
):
    """POST /load_lora_adapter_from_tensors — same as miles _send_lora_params."""
    if unload_first:
        print(f"  Unloading existing adapter '{lora_name}'...")
        resp = requests.post(f"{base_url}/unload_lora_adapter", json={"lora_name": lora_name})

    serialized, cuda_refs = serialize_named_tensors(named_tensors)
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
    del cuda_refs  # Safe to free after server responded
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
        lora_a = torch.randn(rank, in_features, dtype=param.dtype, device="cpu") * 0.05
        lora_b = torch.randn(out_features, rank, dtype=param.dtype, device="cpu") * 0.05
        lora_tensors.append((f"base_model.model.{base_key}.lora_A.default.weight", lora_a))
        lora_tensors.append((f"base_model.model.{base_key}.lora_B.default.weight", lora_b))

    return lora_tensors


def main():
    parser = argparse.ArgumentParser(description="Reproduce garbage output with base + LoRA weight sync")
    parser.add_argument("--model-path", type=str, default="/root/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--prompt", type=str, default="Which is bigger, 9.9 or 9.11")
    parser.add_argument("--skip-base-sync", action="store_true", help="Skip base weight sync, only send LoRA")
    parser.add_argument(
        "--mimic-colocate",
        action="store_true",
        help="Full colocate lifecycle (split resume: weights first, then kv+cuda)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = build_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    target_modules = lora_config["target_modules"]
    url = args.base_url
    prompt = args.prompt

    # Preload weights
    base_tensors = [(name, param.data.cpu()) for name, param in model.named_parameters()] if not args.skip_base_sync else []
    lora_tensors = create_random_lora_weights(model, target_modules, rank=args.lora_rank)

    def send_all_base():
        chunk_size = 50
        for i in range(0, len(base_tensors), chunk_size):
            chunk = base_tensors[i : i + chunk_size]
            send_base_weights(url, chunk, version="1")
        print(f"  Sent {len(base_tensors)} base params")

    def send_all_lora():
        send_lora_weights(url, lora_tensors, lora_config, lora_name="miles_lora")
        print(f"  Sent {len(lora_tensors)} LoRA params")

    # --- Baseline ---
    print("\n=== Baseline inference (original server weights) ===")
    try_inference(url, prompt, "baseline")

    if args.mimic_colocate:
        print("\n=== Colocate flow: release(ALL) → resume(weights) → update → resume(kv+cuda) → inference ===")
        release_memory_occupation(url, tags=["weights", "kv_cache", "cuda_graph"])
        resume_memory_occupation(url, tags=["weights"])
        pause_generation(url)
        flush_cache(url)
        send_all_base()
        send_all_lora()
        continue_generation(url)
        resume_memory_occupation(url, tags=["kv_cache", "cuda_graph"])
        try_inference(url, prompt, "[base only]")
        try_inference(url, prompt, "[with LoRA]", lora_name="miles_lora")
    else:
        send_all_base()
        send_all_lora()
        try_inference(url, prompt, "[base only]")
        try_inference(url, prompt, "[with LoRA]", lora_name="miles_lora")

    del model


if __name__ == "__main__":
    main()
