from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
output_dir = "/root/lora_model/qwen3-30b"

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", 
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
peft_model.to(torch.bfloat16)

peft_model.save_pretrained(output_dir)
print(f"LoRA adapter saved to {output_dir}")