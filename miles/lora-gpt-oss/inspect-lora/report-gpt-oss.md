# GPT-OSS-20B LoRA Adapter Structure Report

## Overview

- **Base model**: openai/gpt-oss-20b (MoE architecture, 32 experts per layer, 24 layers)
- **Total adapters found**: 140
- **Adapters with analyzable safetensors**: ~110 (some had download timeouts or no safetensors)
- **Adapters without safetensors files**: ~30 (checkpoint-only or incomplete uploads)

## Model Architecture Reference (gpt-oss-20b)

| Component | Dimensions |
|-----------|-----------|
| Hidden size | 2880 |
| Q proj output / head dim | 4096 (likely 32 heads × 128 head_dim) |
| K/V proj output | 512 (GQA, likely 4 KV heads × 128) |
| O proj | (4096 → 2880) |
| Num experts | 32 |
| Expert intermediate | 2880 (gate_up_proj fused to 5760 = 2 × 2880) |
| Num layers | 24 |
| MLP router | (2880 → 32) |
| Expert weight names | `w1` (gate_proj), `w2` (down_proj), `w3` (up_proj) — fused format |
| Expert alt names | `gate_up_proj` (fused gate+up = 5760), `down_proj` (2880) |

## Adapter Structure Patterns

### Pattern A: Attention-Only LoRA — q/k/v/o_proj (91+ adapters)

**This is by far the most common pattern**, covering the large majority of adapters.

**Representative groups**:
- 30 adapters at rank=32, fp32 (Group 1: EmilRyd series, GatlingPeaShooter, PhaaNe, yobro4619, etc.)
- 22 adapters at rank=128, bf16 (Group 3: besimray/benchmark, johngreendr1, rayonlabs/benchmark series)
- 20 adapters at rank=16, fp32 (Group 4: JustinAngel, a3ilab, sahilmob, agency42, etc.)
- 19 adapters at rank=8, fp32 (Group 5: BenTouss, cakeng, OmnisAI, sik247, etc.)
- 1 adapter at rank=128, fp32 (LordDenihol)
- 1 adapter at rank=8, bf16 (ZeroAgency)

**Per-layer structure** (example with rank=32):

| Tensor Name | Shape | Dtype |
|------------|-------|-------|
| `self_attn.q_proj.lora_A.weight` | (32, 2880) | fp32 |
| `self_attn.q_proj.lora_B.weight` | (4096, 32) | fp32 |
| `self_attn.k_proj.lora_A.weight` | (32, 2880) | fp32 |
| `self_attn.k_proj.lora_B.weight` | (512, 32) | fp32 |
| `self_attn.v_proj.lora_A.weight` | (32, 2880) | fp32 |
| `self_attn.v_proj.lora_B.weight` | (512, 32) | fp32 |
| `self_attn.o_proj.lora_A.weight` | (32, 4096) | fp32 |
| `self_attn.o_proj.lora_B.weight` | (2880, 32) | fp32 |

**Notes**: 192 tensors total (24 layers × 8 tensors). No expert FFN LoRA.

---

### Pattern A2: Attention-Only LoRA — q/v only (2 adapters)

**Adapters**: `sumomo192300/gpt-oss-20b-multilingual-reasoner_ari`, `sumomo192300/gpt-oss-20b-multilingual-reasoner_kch`

Same as Pattern A but only targets `q_proj` and `v_proj` (no k_proj, no o_proj). 144 tensors, rank=8, fp32.

---

### Pattern A3: Attention-Only LoRA — q/v only (1 adapter)

**Adapter**: `arunimas1107/gpt-oss-medical`

Only `q_proj` and `v_proj`, rank=16, fp32. 96 tensors.

---

### Pattern B: Attention + Router (modules_to_save) LoRA (2 adapters)

**Adapters**: `cakeng/gpt-oss-20b-lora-dolmino-wiki-router-12000-steps`, `cakeng/gpt-oss-20b-lora-dolmino-wiki-router-2000-steps`

**Per-layer structure** (rank=8, fp32):

| Tensor Name | Shape | Dtype | Notes |
|------------|-------|-------|-------|
| `mlp.router.weight` | (32, 2880) | fp32 | **Full router weight saved** (not LoRA) |
| `mlp.router.bias` | (32,) | fp32 | **Full router bias saved** |
| `self_attn.q_proj.lora_A.weight` | (8, 2880) | fp32 | Standard LoRA |
| `self_attn.q_proj.lora_B.weight` | (4096, 8) | fp32 | |
| `self_attn.k_proj.lora_A/B.weight` | standard | fp32 | |
| `self_attn.v_proj.lora_A/B.weight` | standard | fp32 | |
| `self_attn.o_proj.lora_A/B.weight` | standard | fp32 | |

**Notes**: 240 tensors. Saves the full MoE router weights as `modules_to_save` alongside attention LoRA. **No expert FFN LoRA.**

---

### Pattern C: Attention + Shared Expert LoRA (fused gate_up_proj style) (~15 adapters)

**Adapters** (various subgroups):
- Group 6 (3 adapters, rank=16/512): JustinAngel/sycophancy, TARARARAK, garethpaul
- Group 7 (3 adapters, rank=64/2048): finalform/foam, hchunnn, paperboygold
- Group 9 (2 adapters, rank=8/256): Long16, cuongdk253
- Additional: cuongdk253/gptoss-raft, finalform/foamGPT-oss-20B, foamGss-20B-trl, kattyan, michaelwaves, unlimitedbytes, yiwenX, InstalilyAI, solarmar

**Per-layer structure** (example from Long16, rank=8 attn + rank=256 expert):

| Tensor Name | Shape | Dtype | Notes |
|------------|-------|-------|-------|
| `mlp.experts.base_layer.lora_A.weight` | (256, 2880) | fp32 | **Fused gate_up_proj LoRA A** (shared across all 32 experts) |
| `mlp.experts.base_layer.lora_B.weight` | (5760, 256) | fp32 | **Fused gate_up_proj LoRA B** (5760 = 2×2880, shared) |
| `mlp.experts.lora_A.weight` | (256, 2880) | fp32 | **down_proj LoRA A** (shared across all 32 experts) |
| `mlp.experts.lora_B.weight` | (2880, 256) | fp32 | **down_proj LoRA B** (shared) |
| `self_attn.q_proj.lora_A.weight` | (8, 2880) | fp32 | Standard attention LoRA |
| `self_attn.q_proj.lora_B.weight` | (4096, 8) | fp32 | |
| `self_attn.k_proj.lora_A/B` | standard | fp32 | |
| `self_attn.v_proj.lora_A/B` | standard | fp32 | |
| `self_attn.o_proj.lora_A/B` | standard | fp32 | |

**Expert LoRA key details**:
- Uses `experts.base_layer` for the fused gate_up_proj (output dim 5760 = gate+up concatenated)
- Uses `experts` (without `.base_layer`) for down_proj (output dim 2880)
- **Shared across ALL 32 experts** — same LoRA A/B pair for every expert
- Expert LoRA rank is typically higher than attention rank (e.g., 256 vs 8)
- Some adapters only apply expert LoRA to **select layers** (e.g., layers 0, 7, 10, 15, 23) rather than all 24
- The fused gate_up_proj naming means gate_proj and up_proj weights are concatenated: `[gate_proj | up_proj]` with dim 5760

**Variant — selective layer expert LoRA** (e.g., JustinAngel/sycophancy):
- Expert LoRA only on layers 7, 15, 23 (not all 24 layers)
- Other layers have attention-only LoRA

---

### Pattern D: Attention + Shared 3D Expert LoRA (w1/w2/w3 style) (1 adapter)

**Adapter**: `melodyhorse/gpt-oss-20b-triviaqa-rl`

**Config**: `target_modules: "all-linear"`, rank=32, fp32, 338 tensors

**Per-layer structure**:

| Tensor Name | Shape | Dtype | Notes |
|------------|-------|-------|-------|
| `mlp.experts.w1.lora_A.weight` | **(1, 32, 2880)** | fp32 | gate_proj: lora_A **shared** (dim0=1, broadcast) |
| `mlp.experts.w1.lora_B.weight` | **(32, 2880, 32)** | fp32 | gate_proj: lora_B **per-expert** (dim0=32) |
| `mlp.experts.w2.lora_A.weight` | **(32, 32, 2880)** | fp32 | down_proj: lora_A **per-expert** (dim0=32) |
| `mlp.experts.w2.lora_B.weight` | **(1, 2880, 32)** | fp32 | down_proj: lora_B **shared** (dim0=1, broadcast) |
| `mlp.experts.w3.lora_A.weight` | **(1, 32, 2880)** | fp32 | up_proj: lora_A **shared** (dim0=1, broadcast) |
| `mlp.experts.w3.lora_B.weight` | **(32, 2880, 32)** | fp32 | up_proj: lora_B **per-expert** (dim0=32) |
| `attn.q_proj.lora_A.weight` | (32, 2880) | fp32 | Standard 2D LoRA |
| `attn.q_proj.lora_B.weight` | (4096, 32) | fp32 | |
| `attn.k_proj.lora_A/B` | standard 2D | fp32 | |
| `attn.v_proj.lora_A/B` | standard 2D | fp32 | |
| `attn.o_proj.lora_A/B` | standard 2D | fp32 | |

**Expert LoRA key details**:
- Uses 3D tensors — same partially-shared pattern as Qwen3 Pattern A
- `w1/w3` (gate/up): lora_A shared (dim0=1), lora_B per-expert (dim0=32)
- `w2` (down): lora_A per-expert (dim0=32), lora_B shared (dim0=1)
- Names use `w1/w2/w3` (not `gate_proj/up_proj/down_proj`) — indicates fused expert weights
- Attention uses `attn.` prefix (not `self_attn.`) — likely a different modeling library
- All 24 layers have both attention and expert LoRA

---

### Pattern E: Full Model Weights (not LoRA) (1 adapter)

**Adapter**: `aifeifei798/QiMing-Janus-20B`

277 tensors, bf16. Contains full model weights (not LoRA adapters) — `self_attn.q_proj.weight`, `self_attn.q_proj.bias`, `mlp.experts.gate_up_proj_bias`, `mlp.router.weight/bias`, `input_layernorm.weight`, etc. Only covers 19 of 24 layers.

---

### Pattern F: Multi-file LoRA (multiple safetensors files) (1 adapter)

**Adapter**: `AbstractPhil/mirel-gpt-oss-20b`

2112 tensors across multiple safetensors files. All attention-only LoRA (q/k/v/o_proj) at rank=16, fp32. Multiple files likely from multi-adapter (e.g., 11 LoRA adapters saved together: 2112 / 192 ≈ 11).

---

### Empty/Incomplete (2 adapters)

**Adapters**: `AkiK/gpt-oss-20b-cpt_sft_combo-merged-16bit`, `SutskeverFanBoy/oss20b-turkish-qlora-backup`

Safetensors files exist but contain 0 tensors.

---

## Expert Layer LoRA Summary

| Aspect | Pattern C (Shared Fused) | Pattern D (Shared 3D) | No Expert (A/B) |
|--------|-------------------------|----------------------|-----------------|
| Expert weight naming | `experts.base_layer` (gate_up), `experts` (down) | `experts.w1/w2/w3` | N/A |
| LoRA tensor shape | 2D: (rank, dim) | 3D: (N_experts, dim, rank) | N/A |
| Fused? | Yes: gate_up_proj fused (5760) | No: separate w1/w2/w3 | N/A |
| Shared across experts? | Yes: fully shared (single LoRA pair) | Partially: one of A/B is shared (dim0=1) | N/A |
| Expert tensors per layer | 4 (2 projections × 2) | 6 (3 projections × 2) | 0 |
| Layer coverage | Sometimes selective (e.g., layers 0,7,10,15,23) | All 24 layers | N/A |
| Expert LoRA rank | 256-2048 (much larger than attn) | 32 (same as attn) | N/A |
| Attention rank | 8-64 | 32 | 8-128 |
| Adapter count | ~15 | 1 | ~91 |

## Rank Distribution (analyzable adapters)

| Rank | Count | Notes |
|------|-------|-------|
| 8 | ~19 | Common for attention-only |
| 16 | ~20 | Common for attention-only |
| 32 | ~30 | Most popular for attention-only |
| 64 | ~3 | Less common |
| 128 | ~23 | Popular for benchmark/rayonlabs series |
| 8+256 | ~10 | Attention rank 8 + expert rank 256 |
| 16+512 | ~3 | Attention rank 16 + expert rank 512 |
| 64+2048 | ~3 | Attention rank 64 + expert rank 2048 |

## Key Observations

1. **Vast majority (65%+) of adapters are attention-only** — they don't include any expert/FFN LoRA at all.
2. **When expert LoRA is included**, the dominant approach is **fully shared** across all 32 experts (Pattern C), using fused gate_up_proj naming.
3. **Expert LoRA rank is typically 8-32× higher** than attention LoRA rank (e.g., 256 vs 8), reflecting that a single LoRA pair is shared across all experts.
4. **Selective layer expert LoRA** is sometimes used — expert LoRA applied to only a subset of layers (e.g., 0, 7, 10, 15, 23) while attention LoRA covers all 24 layers.
5. **The 3D partially-shared expert LoRA pattern** (Pattern D) appears only once, via `melodyhorse/gpt-oss-20b-triviaqa-rl`, using a different modeling framework (`attn.` vs `self_attn.`).
6. **No per-expert individual LoRA** was observed (unlike Qwen3-30B-A3B which had one adapter with separate LoRA per expert).
7. **Router LoRA** was rare — only 2 adapters (cakeng router variants) saved full router weights, and no adapter used LoRA on the router.
