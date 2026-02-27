import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from peft import LoraConfig, get_peft_model


class GptOssExpertsAsLinear(nn.Module):
    """
    Drop-in replacement for GptOssExperts that splits the batched parameter
    tensors into individual nn.Linear modules per expert, so peft can target
    gate_up_proj and down_proj by name.
    """

    def __init__(self, base: GptOssExperts):
        super().__init__()
        self.alpha = base.alpha
        self.limit = base.limit
        self.hidden_size = base.hidden_size
        self.num_experts = base.num_experts

        experts = []
        for i in range(base.num_experts):
            e = nn.Module()
            # Original param gate_up_proj[i]: [h, 2d], used as x @ W
            # nn.Linear stores weight as [out, in], so weight = W.T
            e.gate_up_proj = nn.Linear(base.hidden_size, 2 * base.expert_dim)
            e.gate_up_proj.weight.data = base.gate_up_proj[i].T.contiguous()
            e.gate_up_proj.bias.data = base.gate_up_proj_bias[i].contiguous()

            e.down_proj = nn.Linear(base.expert_dim, base.hidden_size)
            e.down_proj.weight.data = base.down_proj[i].T.contiguous()
            e.down_proj.bias.data = base.down_proj_bias[i].contiguous()

            experts.append(e)
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        batch = hidden_states.shape[0]
        flat = hidden_states.reshape(-1, self.hidden_size)
        n = routing_weights.shape[1]

        out = torch.zeros_like(flat)
        with torch.no_grad():
            mask = F.one_hot(router_indices, num_classes=n + 1).permute(2, 1, 0)
            hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        for idx in hits:
            eidx = idx[0].item()
            if eidx == n:
                continue
            with torch.no_grad():
                _, tok = torch.where(mask[eidx])
            x = flat[tok]

            gate_up = self.experts[eidx].gate_up_proj(x)
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            mid = (up + 1) * gate * torch.sigmoid(gate * self.alpha)

            y = self.experts[eidx].down_proj(mid)
            out.index_add_(0, tok, (y * routing_weights[tok, eidx, None]).to(flat.dtype))

        return out.view(batch, -1, self.hidden_size)


def replace_experts_with_linear(model):
    for name, module in list(model.named_modules()):
        if not isinstance(module, GptOssExperts):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], GptOssExpertsAsLinear(module))
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

model_name = "openai/gpt-oss-20b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Convert batched params → nn.Linear so peft can find them by name
replace_experts_with_linear(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "gate_proj",
        "up_proj"
        "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules="all-linear",
#     target_parameters=[
#         "7.mlp.experts.gate_up_proj",
#         "7.mlp.experts.down_proj",
#         "15.mlp.experts.gate_up_proj",
#         "15.mlp.experts.down_proj",
#         "23.mlp.experts.gate_up_proj",
#         "23.mlp.experts.down_proj",
#     ],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
lora_model.to(torch.bfloat16)

output_dir = "/root/lora_model/gpt-oss-20b"
lora_model.save_pretrained(output_dir)
print(f"LoRA adapter saved to {output_dir}")
