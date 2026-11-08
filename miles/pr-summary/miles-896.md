# Miles #896: Faster LoRA weight synchronization

## Abstract

[PR #896](https://github.com/radixark/miles/pull/896) removes redundant memory backup/restore work in colocated LoRA RL, reducing Qwen2.5-3B weight-update time from ~5 s to ~0.3 s.

## Background

SGLang previously backed up frozen base weights to CPU during offload and restored them to GPU for rollout. Adapters already used CUDA IPC, but accessing them required a trainer-wide `resume()` / `pause()` cycle.

That cycle performed synchronous CPU↔GPU copies, pinned-memory allocation/free, and GPU mapping/unmapping for each tracked allocation. Even a 3B BF16 model has ~6 GB of base weights. Disk loading occurred at initialization; the repeated overhead was memory backup/restore.

## Design

- Split weight export by `weight_type="base"` or `"lora"`, then send through `_send_base_params()` and `_send_lora_params()` over CUDA IPC.
- Repopulate SGLang's base weights from the trainer, removing its separate CPU backup.
- Keep adapter buffers resident in separate GPU memory regions; pause only the trainer's `"default"` region and remove the extra resume/pause around synchronization.

```text
Base: trainer backup → CUDA export buffers → CUDA IPC → SGLang
LoRA: resident GPU adapter buffers → CUDA IPC → SGLang
```

Trainer-side base weights can still require CPU→GPU staging before IPC transfer.

## Benchmark result

[Reported results](https://github.com/radixark/miles/pull/896#issuecomment-4203897511) for **Qwen2.5-3B, colocated LoRA RL**:

| Metric | Before | After | Improvement |
| --- | --- | --- | --- |
| Weight update (`perf/update_weights_time`) | ~5 s | ~0.3 s | ~17× faster |
| Step (`perf/step_time`) | ~30 s | ~21 s | ~30% less time |

Weight-update time now matches colocated full-parameter RL. Its timer includes trainer memory management, so the speedup measures the whole update operation. Step time includes training plus waiting. No per-operation breakdown or disaggregated-mode measurements were provided.

## Downstream impact / feedback

- **SGLang correctness:** related fixes [#22169](https://github.com/sgl-project/sglang/pull/22169) and [#22846](https://github.com/sgl-project/sglang/pull/22846) expose base-layer biases through LoRA wrappers, ensuring base-weight reloads include them.
- **Follow-up opportunity:** [review feedback](https://github.com/radixark/miles/pull/896#discussion_r3037502583) suggested skipping repeated frozen base-weight synchronization when rollout offloading is disabled and the weights remain resident.
