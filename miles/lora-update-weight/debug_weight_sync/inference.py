"""
Quick inference test against a running SGLang server.
Usage:
  python scripts/repro/test_inference.py
  python scripts/repro/test_inference.py --lora miles_lora
  python scripts/repro/test_inference.py --prompt "Explain gravity in one sentence."
"""

import argparse

import requests


def generate(base_url: str, prompt: str, max_tokens: int = 128, temperature: float = 0, lora: str | None = None):
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature},
    }
    if lora:
        payload["lora_path"] = lora
    resp = requests.post(f"{base_url}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--lora", default=None, help="LoRA adapter name, e.g. miles_lora")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else [
        "Which is bigger, 9.9 or 9.11?",
    ]

    mode = f"LoRA={args.lora}" if args.lora else "base"
    print(f"Server: {args.base_url}  Mode: {mode}\n")

    for p in prompts:
        output = generate(args.base_url, p, args.max_tokens, args.temperature, args.lora)
        print(f"[Prompt]  {p}")
        print(f"[Output]  {output}")
        print()


if __name__ == "__main__":
    main()
