from safetensors import safe_open

path = "/root/lora_model/qwen1.5-moe-a2.7b/adapter_model.safetensors"

with safe_open(path, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        print(f"{k:80s} {str(t.shape):30s} {t.dtype}")
