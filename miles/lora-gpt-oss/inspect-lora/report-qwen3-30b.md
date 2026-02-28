# Qwen3-30B-A3B LoRA Adapter Structure Report

## Overview

- **Base model**: Qwen/Qwen3-30B-A3B (MoE architecture, 128 experts per layer, 48 layers)
- **Total adapters found**: 31
- **Adapters with safetensors files**: 12 (39%)
- **Adapters without safetensors files**: 19 (61%) — these contain only checkpoint directories, adapter configs, or tokenizer files

## Model Architecture Reference (Qwen3-30B-A3B)

| Component | Dimensions |
|-----------|-----------|
| Hidden size | 2048 |
| Intermediate size (expert) | 768 |
| Num attention heads | 32 (q_proj output: 4096 with head_dim=128) |
| Num KV heads | 4 (k/v_proj output: 512) |
| Num experts | 128 |
| Num layers | 48 |
| MLP gate (router) | (2048 → 128) |

## Adapter Structure Patterns

### Pattern A: Shared 3D Expert LoRA + Attention (3 adapters)

**Adapters**:
- `abugoot-primeintellect/bioreasoning-qwen3-30ba3b-sft-20260218-ckpt-1500` (rank=32, fp32)
- `abugoot-primeintellect/bioreasoning-qwen3-30ba3b-sft-20260218-ckpt-3000` (rank=32, fp32)
- `k-l-lambda/qwen3-30b-a3b-r2e-gym-sft` (rank=16, fp32)

**Config**: `target_modules: "all-linear"`, all 48 layers, 674 tensors per adapter

**Per-layer structure** (example with rank=32):

| Tensor Name | Shape | Dtype | Notes |
|------------|-------|-------|-------|
| `mlp.experts.w1.lora_A.weight` | (1, 32, 2048) | fp32 | **Shared** across 128 experts (dim0=1, broadcast) |
| `mlp.experts.w1.lora_B.weight` | (128, 768, 32) | fp32 | **Per-expert** (dim0=128) |
| `mlp.experts.w2.lora_A.weight` | (128, 32, 768) | fp32 | **Per-expert** (dim0=128) |
| `mlp.experts.w2.lora_B.weight` | (1, 2048, 32) | fp32 | **Shared** across 128 experts (dim0=1, broadcast) |
| `mlp.experts.w3.lora_A.weight` | (1, 32, 2048) | fp32 | **Shared** across 128 experts (dim0=1, broadcast) |
| `mlp.experts.w3.lora_B.weight` | (128, 768, 32) | fp32 | **Per-expert** (dim0=128) |
| `self_attn.q_proj.lora_A.weight` | (32, 2048) | fp32 | Standard 2D LoRA |
| `self_attn.q_proj.lora_B.weight` | (4096, 32) | fp32 | Standard 2D LoRA |
| `self_attn.k_proj.lora_A.weight` | (32, 2048) | fp32 | Standard 2D LoRA |
| `self_attn.k_proj.lora_B.weight` | (512, 32) | fp32 | Standard 2D LoRA |
| `self_attn.v_proj.lora_A.weight` | (32, 2048) | fp32 | Standard 2D LoRA |
| `self_attn.v_proj.lora_B.weight` | (512, 32) | fp32 | Standard 2D LoRA |
| `self_attn.o_proj.lora_A.weight` | (32, 4096) | fp32 | Standard 2D LoRA |
| `self_attn.o_proj.lora_B.weight` | (2048, 32) | fp32 | Standard 2D LoRA |

**Expert LoRA key insight**: Uses **3D tensors** where:
- `w1` (gate_proj equivalent): lora_A is **shared** (dim0=1), lora_B is **per-expert** (dim0=128)
- `w2` (down_proj equivalent): lora_A is **per-expert** (dim0=128), lora_B is **shared** (dim0=1)
- `w3` (up_proj equivalent): same as w1 — lora_A **shared**, lora_B **per-expert**
- The naming `w1/w2/w3` (vs `gate_proj/up_proj/down_proj`) indicates these are **fused** expert weights
- The sharing pattern means: for w1/w3, all experts share the same input projection (lora_A) but have distinct output projections (lora_B). For w2, vice versa.

---

### Pattern B: Per-Expert Individual LoRA (1 adapter)

**Adapters**:
- `chenrm/qwen3-30b-a3b-abliterated-lora` (rank=4, bf16)

**Config**: `target_modules: [gate, q_proj, v_proj, down_proj, up_proj, o_proj, k_proj, gate_proj]`, `modules_to_save: [embed_tokens, q_norm, k_norm, input_layernorm, post_attention_layernorm, norm, lm_head]`, all 48 layers, 37,441 tensors

**Per-layer structure** (example for expert 0):

| Tensor Name | Shape | Dtype | Notes |
|------------|-------|-------|-------|
| `mlp.experts.0.gate_proj.lora_A.weight` | (4, 2048) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.experts.0.gate_proj.lora_B.weight` | (768, 4) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.experts.0.up_proj.lora_A.weight` | (4, 2048) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.experts.0.up_proj.lora_B.weight` | (768, 4) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.experts.0.down_proj.lora_A.weight` | (4, 768) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.experts.0.down_proj.lora_B.weight` | (2048, 4) | bf16 | Standard 2D LoRA, per-expert |
| `mlp.gate.lora_A.weight` | (4, 2048) | bf16 | Router LoRA |
| `mlp.gate.lora_B.weight` | (128, 4) | bf16 | Router LoRA |
| `self_attn.{q,k,v,o}_proj.lora_{A,B}.weight` | standard 2D | bf16 | Same as Pattern A |
| `input_layernorm.weight` | (2048,) | bf16 | Saved (not LoRA) |

**Key differences from Pattern A**:
- Expert weights use standard 2D LoRA tensors — **separate LoRA A/B pair per expert** (128 × 3 projections × 2 = 768 expert LoRA tensors per layer)
- Naming uses `gate_proj/up_proj/down_proj` (not fused `w1/w2/w3`)
- Also includes LoRA on the router (`mlp.gate`)
- Additionally saves non-LoRA modules (layernorms, embeddings, lm_head)
- Very large number of tensors (37,441) due to per-expert approach with 128 experts

---

### Pattern C: Attention + Router LoRA, No Expert LoRA (3 adapters)

**Adapters**:
- `debaterhub/Bohr-LoRA-v1-Qwen3-30B` (rank=64, fp32)
- `debaterhub/Einstein-LoRA-v2-Qwen3-30B` (rank=64, fp32)
- `debaterhub/AntiSycophancy-LoRA-Qwen3-30B` (rank=64, bf16)

**Config**: `target_modules: [v_proj, q_proj, o_proj, mlp.gate, k_proj]`, all 48 layers, 480 tensors

**Per-layer structure**:

| Tensor Name | Shape | Dtype |
|------------|-------|-------|
| `mlp.gate.lora_A.weight` | (64, 2048) | fp32/bf16 |
| `mlp.gate.lora_B.weight` | (128, 64) | fp32/bf16 |
| `self_attn.q_proj.lora_A.weight` | (64, 2048) | fp32/bf16 |
| `self_attn.q_proj.lora_B.weight` | (4096, 64) | fp32/bf16 |
| `self_attn.k_proj.lora_A.weight` | (64, 2048) | fp32/bf16 |
| `self_attn.k_proj.lora_B.weight` | (512, 64) | fp32/bf16 |
| `self_attn.v_proj.lora_A.weight` | (64, 2048) | fp32/bf16 |
| `self_attn.v_proj.lora_B.weight` | (512, 64) | fp32/bf16 |
| `self_attn.o_proj.lora_A.weight` | (64, 4096) | fp32/bf16 |
| `self_attn.o_proj.lora_B.weight` | (2048, 64) | fp32/bf16 |

**Notes**: Targets attention projections AND the MoE router (`mlp.gate`), but **no expert FFN LoRA**.

---

### Pattern D: Attention-Only LoRA, No Expert LoRA (5 adapters)

**Adapters**:
- `debaterhub/debate-orpo-iter12` (rank=32, bf16)
- `debaterhub/sentence-selection-orpo-lora` (rank=32, bf16)
- `debaterhub/sentence-selection-orpo-v3` (rank=32, fp32)
- `matt4512/Qwen3-30B-A3B-linkedin-comments` (rank=8, fp32)

**Config**: `target_modules: [q_proj, k_proj, v_proj, o_proj]`, all 48 layers, 384 tensors

**Per-layer structure**:

| Tensor Name | Shape | Dtype |
|------------|-------|-------|
| `self_attn.q_proj.lora_A.weight` | (rank, 2048) | varies |
| `self_attn.q_proj.lora_B.weight` | (4096, rank) | varies |
| `self_attn.k_proj.lora_A.weight` | (rank, 2048) | varies |
| `self_attn.k_proj.lora_B.weight` | (512, rank) | varies |
| `self_attn.v_proj.lora_A.weight` | (rank, 2048) | varies |
| `self_attn.v_proj.lora_B.weight` | (512, rank) | varies |
| `self_attn.o_proj.lora_A.weight` | (rank, 4096) | varies |
| `self_attn.o_proj.lora_B.weight` | (2048, rank) | varies |

**Notes**: Only targets attention projections. **No MoE expert or router LoRA at all.**

---

### Pattern E: Prefix Tuning (1 adapter)

**Adapter**: `debaterhub/prefix-einstein`

1 tensor only — this is **prefix tuning**, not LoRA.

---

### No Safetensors (19 adapters)

These adapters don't contain `adapter_model.safetensors` files directly. They typically contain:
- `adapter_config.json` only (weights in checkpoint subdirectories or not uploaded)
- Checkpoint directories (`checkpoint-N/`) with training state

**Adapters**: Achilles1089/achilles-30b-adapter, GhostNetworkUser/KumpelAi, MastermanF/Turanllm10052025, Rorical/qode-30b-lora, abugoot-primeintellect/bioreasoning-qwen3-30ba3b-sft-20260218-ckpt-final, airesearch/Qwen3-30B-A3B-alpaca-th-52k-dolly-th-15k-wangchan-instruct (×3 seeds), airesearch/Qwen3-30B-A3B-medqa (×3 seeds), danghuyhoang/qwen3-30b-vietnamese-instruct, debaterhub/debate-sft-group-a-opus-distilled, dgonier/iter3-grpo-lora-groupA-D, mohamedrayyan/dhivehi-news-lora-qwen3-30b, shuttleai/shuttle-3.5-moe-ckpts

---

## Expert Layer LoRA Summary

| Aspect | Pattern A (Shared 3D) | Pattern B (Per-Expert) | Patterns C/D (No Expert) |
|--------|----------------------|----------------------|-------------------------|
| Expert weight naming | `w1/w2/w3` (fused) | `gate_proj/up_proj/down_proj` (unfused) | N/A |
| LoRA tensor shape | 3D: `(N, out, rank)` | 2D: `(out, rank)` per expert | N/A |
| Shared across experts? | Partially: lora_A or lora_B has dim0=1 (broadcast) | No: fully independent per expert | N/A |
| Expert tensors per layer | 6 (3 projections × 2) | 768 (128 experts × 3 projections × 2) | 0 |
| Total tensors (all layers) | 674 | 37,441 | 384-480 |
| Router (`mlp.gate`) LoRA? | No (but config says "all-linear") | Yes | Pattern C: Yes; Pattern D: No |
| Rank used | 16-32 | 4 | 8-64 |
| Dtype | fp32 | bf16 | fp32 or bf16 |

## Rank Distribution

| Rank | Count | Adapters |
|------|-------|----------|
| 4 | 1 | chenrm/abliterated-lora |
| 8 | 1 | matt4512/linkedin-comments |
| 16 | 1 | k-l-lambda/r2e-gym-sft |
| 32 | 5 | abugoot (×2), debaterhub-orpo (×2), debaterhub-sentence-v3 |
| 64 | 3 | debaterhub Bohr/Einstein/AntiSycophancy |
