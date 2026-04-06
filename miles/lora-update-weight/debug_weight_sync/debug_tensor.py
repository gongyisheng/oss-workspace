import hashlib
from pathlib import Path

import torch

DEFAULT_WEIGHT_DIR = "/root/debug-weight"


def _hash_cpu(t: torch.Tensor) -> str:
    t = t.detach().contiguous().view(-1).view(torch.int8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def tensor_stats(name: str, t: torch.Tensor) -> str:
    flat = t.detach().float().flatten()
    h = _hash_cpu(t.cpu())
    return (
        f"{name:60s} | shape={str(list(t.shape)):20s} dtype={str(t.dtype):12s} "
        f"hash={h}  min={flat.min().item(): .6e}  max={flat.max().item(): .6e}  "
        f"mean={flat.mean().item(): .6e}  std={flat.std().item(): .6e}  "
        f"norm={flat.norm().item(): .6e}"
    )


def dump_model_stats(model, prefix="", pattern=None):
    """Print stats for all parameters in a model.

    Args:
        model: nn.Module or dict of name->tensor
        prefix: optional prefix for print lines
        pattern: optional substring filter on parameter names
    """
    items = model.items() if isinstance(model, dict) else model.named_parameters()
    for name, param in items:
        if pattern and pattern not in name:
            continue
        data = param.data if hasattr(param, "data") else param
        print(f"{prefix}{tensor_stats(name, data)}")


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    """Compare two tensors and report diff stats."""
    diff = (a.detach().float() - b.detach().float()).flatten()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    match = torch.equal(a, b)
    return (
        f"{name:60s} | match={match}  max_abs_diff={max_abs:.6e}  "
        f"mean_abs_diff={mean_abs:.6e}  diff_norm={diff.norm().item():.6e}"
    )


def load_chunks(directory: str) -> dict[str, torch.Tensor]:
    """Load all chunk_*.pt files from a directory into a single dict."""
    tensors = {}
    for p in sorted(Path(directory).glob("chunk_*.pt")):
        tensors.update(torch.load(p, map_location="cpu"))
    return tensors


def load_debug_weights(
    version: str = "v1",
    rank: int = 0,
    weight_dir: str = DEFAULT_WEIGHT_DIR,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load base and lora weights from debug-weight directory.

    Returns:
        (base_weights, lora_weights) as dicts of name -> tensor
    """
    root = Path(weight_dir) / version / f"rank{rank}"
    base = load_chunks(root / "base")
    lora = load_chunks(root / "lora")
    print(f"Loaded {len(base)} base tensors, {len(lora)} lora tensors from {root}")
    return base, lora


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug tensor stats")
    parser.add_argument("--version", default="v1", help="Weight version (default: v1)")
    parser.add_argument("--rank", type=int, default=0, help="Rank (default: 0)")
    parser.add_argument("--weight-dir", default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--pattern", default=None, help="Filter parameter names")
    parser.add_argument("--type", choices=["base", "lora", "all"], default="all")
    args = parser.parse_args()

    base, lora = load_debug_weights(args.version, args.rank, args.weight_dir)

    if args.type in ("base", "all"):
        print(f"\n{'='*40} BASE WEIGHTS {'='*40}")
        dump_model_stats(base, pattern=args.pattern)

    if args.type in ("lora", "all"):
        print(f"\n{'='*40} LORA WEIGHTS {'='*40}")
        dump_model_stats(lora, pattern=args.pattern)
