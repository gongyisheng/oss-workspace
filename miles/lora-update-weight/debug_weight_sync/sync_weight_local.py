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
import glob
import json
import os

import requests
import torch

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
    # flush cache first, same as SGLangEngine.release_memory_occupation
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


def load_base_weights(weight_dir: str) -> list[tuple[str, torch.Tensor]]:
    """Load base weights from chunk .pt files saved by miles."""
    base_dir = os.path.join(weight_dir, "v1", "rank0", "base")
    chunk_files = sorted(glob.glob(os.path.join(base_dir, "chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No base weight chunks found in {base_dir}")

    base_tensors = []
    for path in chunk_files:
        state_dict = torch.load(path, map_location="cpu")
        for name, tensor in state_dict.items():
            base_tensors.append((name, tensor))
    print(f"  Loaded {len(base_tensors)} base params from {len(chunk_files)} chunks")
    return base_tensors


def load_lora_weights(weight_dir: str) -> list[tuple[str, torch.Tensor]]:
    """Load LoRA weights from chunk .pt files saved by miles."""
    lora_dir = os.path.join(weight_dir, "v1", "rank0", "lora")
    chunk_files = sorted(glob.glob(os.path.join(lora_dir, "chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No LoRA weight chunks found in {lora_dir}")

    lora_tensors = []
    for path in chunk_files:
        state_dict = torch.load(path, map_location="cpu")
        for name, tensor in state_dict.items():
            lora_tensors.append((name, tensor))
    print(f"  Loaded {len(lora_tensors)} LoRA params from {len(chunk_files)} chunks")
    return lora_tensors


def try_inference(base_url: str, prompt: str, label: str, lora_name: str | None = None):
    """Try inference and print result. Returns True if request succeeded."""
    try:
        output = generate(base_url, prompt, lora_name=lora_name)
        tag = f"(lora={lora_name})" if lora_name else "(base)"
        print(f"  [{label}] {tag} → {output[:200]}")
        return True
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sync actual weights from disk to SGLang server")
    parser.add_argument("--weight-dir", type=str, default="/root/debug-weight")
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

    lora_config = build_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    url = args.base_url
    prompt = args.prompt

    # Preload weights from disk
    base_tensors = load_base_weights(args.weight_dir) if not args.skip_base_sync else []
    lora_tensors = load_lora_weights(args.weight_dir)

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
        try_inference(url, prompt, "split resume + update (colocate)")
        try_inference(url, prompt, "with LoRA", lora_name="miles_lora")
    else:
        send_all_base()
        send_all_lora()
        try_inference(url, prompt, "after weight update")
        try_inference(url, prompt, "with LoRA", lora_name="miles_lora")


if __name__ == "__main__":
    main()
