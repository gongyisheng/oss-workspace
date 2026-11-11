# Miles #896: Faster LoRA weight synchronization

## Abstract

[PR #896](https://github.com/radixark/miles/pull/896) removes redundant memory backup/restore work in colocated LoRA RL, reducing Qwen2.5-3B weight-update time from ~5 s to ~0.3 s.

## Background

Colocated RL alternates training and rollout on the same GPUs, offloading the inactive engine's tensors to host memory and restoring them for its next phase. In LoRA colocated RL, only adapters change; base weights stay frozen.

Previously, we use `torch_memory_saver` to back up SGLang's frozen base weights to CPU and restore them for each rollout. Adapter synchronization already used CUDA IPC, but required an extra trainer-wide `resume()` / `pause()` cycle, causing substantial overhead. Each cycle repeated synchronous CPU↔GPU copies, pinned-memory allocation/free, and GPU allocation, mapping, and unmapping across tracked allocations.

Also notice that in full param RL, the weight sync time is much lower than lora RL, suggesting that there's potential for improvement.

### torch_memory_saver

`torch_memory_saver` uses `LD_PRELOAD` routes CUDA allocation calls through its custom allocator and manages virtual addresses separately from physical memory. It releases physical memory during `pause()` and maps new memory to the same addresses during `resume()`, allowing existing tensor references and CUDA graphs to remain valid. Optional CPU backups preserve tensor contents.

With CPU backup enabled, the [pause/resume loops](https://github.com/fzyzcjy/torch_memory_saver/blob/d64a639/csrc/core.cpp#L100-L202) and [CUDA helpers](https://github.com/fzyzcjy/torch_memory_saver/blob/d64a639/csrc/utils.h#L106-L135) execute these calls for each selected allocation. Potentially expensive calls are **bold**.

**Pause — offload**

- **`cudaMallocHost(...)`** — allocate pinned CPU backup memory if absent.
- **`cudaMemcpy(..., cudaMemcpyDeviceToHost)`** — synchronously copy GPU data to the CPU backup.
- **`cuMemUnmap(...)`** — unmap GPU memory, keeping the virtual address reserved.
- **`cuMemRelease(...)`** — release the physical GPU allocation.

**Resume — restore**

- `cuDeviceGetAttribute(...)` — query GPUDirect RDMA support for CUDA virtual memory.
- **`cuMemCreate(...)`** — allocate physical GPU memory.
- **`cuMemMap(...)`** — map it to the preserved virtual address.
- **`cuMemSetAccess(...)`** — set GPU access permissions on the mapping.
- **`cudaMemcpy(..., cudaMemcpyHostToDevice)`** — synchronously restore data from the CPU backup.
- **`cudaFreeHost(...)`** — free/unpin the CPU backup.

These repeated operations add substantial overhead. Since base weights remain frozen throughout LoRA training, we can reuse the trainer's base weights in the inference engine through CUDA IPC, avoiding SGLang's separate CPU backup/restore cycle.

## Scope

- **Performance:** reduce colocated LoRA weight-update time by removing unnecessary overhead with reasonable device memory usage increase.
- **Correctness:** ensure the inference engine and training engine always use the same frozen base weight.

## Design

### General Process

Setting: colocate RL

training loop:
- await actor_model.offload_train()
- await rollout_manager.onload_weights.remote()
- await actor_model.update_weights()
- await rollout_manager.onload_kv.remote()

note that only #3 is measured by update_weights timer, #1/2 still uses torch_memory_saver to achieve onload/offload but not tracked by timer

in the following part we only talk about #3 update_weights

**Full-param RL**

- inference engine stop generation and flush KV/prefix cache.
- training engine prepares updated model weights from its cpu backup, staging tensors on GPU and exporting them to HF format
- synchronize model weights through CUDA IPC
- wait until all inference engines are ready
- inference engines resume generation

**LoRA RL before optimization**

- training engine resumes all tracked GPU memory with `torch_memory_saver.resume()` to make adapter parameters accessible
- inference engines stop generation and flush KV/prefix cache.
- training engine exports the updated adapter weights to HF format
- synchronize adapters to all inference engines through CUDA IPC
- wait until all inference engines are ready
- inference engines resume generation.
- training engine pauses all tracked GPU memory again with `torch_memory_saver.pause()`.

**LoRA RL after optimization**

- inference engines stop generation and flush KV/prefix cache.
- training engine prepares frozen base weights from its CPU backup and updated adapters from resident GPU buffers
- synchronize base weights to all inference engines through CUDA IPC
- synchronize adapters through CUDA IPC
- wait until all inference engines are ready
- inference engines resume generation

### Keep Adapter Buffers Resident

We patched megatron side adapter buffers in separate memory-saver regions and pause only the trainer's `"default"` region. Adapters remain on GPU for direct export. (trade-off here: no obvious benefits to offload lora adapter since it's lightweight, easier for us to get its pointer to do cuda ipc, other option is to offload and staging lora weights on gpu again like base weights)

### Frozen Base Weights
miles set base weights required_gard=False, ensure base weights are frozen and only lora weights get updated.
```
base_weight.requires_grad_(False)
lora_A.requires_grad_(True)
lora_B.requires_grad_(True)
```

## Test

[Reported results](https://github.com/radixark/miles/pull/896#issuecomment-4203897511) for Qwen2.5-3B, colocated LoRA RL on 1XH200:
- `perf/update_weights_time` reduced from 5s to 0.3s
- `perf/step_time` reduced from 30s to 21s

ref: https://forge.coreweave.com/wandb/gongyisheng/miles-lora-update-weight-refactory

## Further Improvement
- aware that @nan has some feedback on kimi k3 colocate lora rl, looks this PR causes longer time (600s) to do weight sync, but I cannot reproduce it on my side, happy to help if the problem can be reproduced.