# Miles #896: Faster LoRA weight synchronization

## Abstract

[PR #896](https://github.com/radixark/miles/pull/896) removes redundant memory backup/restore work in colocated LoRA RL, reducing Qwen2.5-3B weight-update time from ~5 s to ~0.3 s.

## Background

SGLang previously backed up frozen base weights to CPU during offload and restored them to GPU for rollout. Adapters already used CUDA IPC, but accessing them required a trainer-wide `resume()` / `pause()` cycle.

That cycle performed synchronous CPU↔GPU copies, pinned-memory allocation/free, and GPU mapping/unmapping for each tracked allocation. Even a 3B BF16 model has ~6 GB of base weights. Disk loading occurred at initialization; the repeated overhead was memory backup/restore.

The PR [pins memory-saver to `d64a639`](https://github.com/radixark/miles/blob/a26ebf6930ede08a6e3c3d159025f23150bab777/docker/Dockerfile#L83). Its [pause/resume loops](https://github.com/fzyzcjy/torch_memory_saver/blob/d64a639/csrc/core.cpp#L100-L203) and [CUDA helpers](https://github.com/fzyzcjy/torch_memory_saver/blob/d64a639/csrc/utils.h#L106-L135) execute these calls in order for each selected allocation with CPU backup enabled.

Cost labels are qualitative, not measured timings: **lightweight** means metadata/query/enqueue work; **potentially heavy** means allocation, mapping, or teardown that can add up across allocations; **heavy copy/wait** depends on bytes transferred or pending GPU work.

| Phase | CUDA API call | Relative cost | Work performed |
| --- | --- | --- | --- |
| Resume, before adapter sync | `cuDeviceGetAttribute(...)` | Lightweight | Query GPUDirect RDMA support for CUDA virtual memory. |
| Resume | `cuMemCreate(...)` | Potentially heavy | Allocate physical GPU memory. |
| Resume | `cuMemMap(...)` | Potentially heavy | Map it into the existing virtual address. |
| Resume | `cuMemSetAccess(...)` | Potentially heavy | Set GPU access permissions on the mapping. |
| Resume | `cudaMemcpy(..., cudaMemcpyHostToDevice)` | Heavy copy | Restore data from the pinned CPU backup. |
| Resume | `cudaFreeHost(...)` | Potentially heavy | Free/unpin the CPU backup. |
| Pause, after adapter sync | `cudaMallocHost(...)` | Potentially heavy | Allocate/pin a new CPU backup if absent. |
| Pause | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | Heavy copy | Copy GPU data back to CPU. |
| Pause | `cuMemUnmap(...)` | Potentially heavy | Unmap GPU memory, retaining its virtual address reservation. |
| Pause | `cuMemRelease(...)` | Potentially heavy | Release the physical GPU allocation. |

GPU memory is allocated with `cuMemCreate` in this cycle; the host backup uses `cudaMallocHost`. The loops use synchronous `cudaMemcpy`, with no explicit `cudaDeviceSynchronize()` call. `cuMemAddressReserve` belongs to initial allocation; `cudaDeviceSynchronize` and `cuMemAddressFree` appear in final allocation teardown, outside these loops. Disabling CPU backup skips the host allocation/free and copy calls, while GPU mapping operations still occur.

## Design

- Split weight export by `weight_type="base"` or `"lora"`, then send through `_send_base_params()` and `_send_lora_params()` over CUDA IPC.
- Repopulate SGLang's base weights from the trainer, removing its separate CPU backup.
- Keep adapter buffers resident in separate GPU memory regions; pause only the trainer's `"default"` region and remove the extra resume/pause around synchronization.

```text
Base: trainer backup → CUDA export buffers → CUDA IPC → SGLang
LoRA: resident GPU adapter buffers → CUDA IPC → SGLang
```

Trainer-side base weights can still require CPU→GPU staging before IPC transfer.

**CUDA IPC operations**

Miles packs tensors into `FlattenedTensorBucket` and serializes them with SGLang's `MultiprocessingSerializer` / PyTorch `ForkingPickler`. For the standard CUDA allocator in [PyTorch 2.9.1, specified by the companion SGLang branch](https://github.com/sgl-project/sglang/blob/96e4d0500ff7447bbd6ca082d4cc1d8e8436dec7/python/pyproject.toml#L66), the main IPC calls are:

| Stage | CUDA API calls | Relative cost / behavior |
| --- | --- | --- |
| Export buffer handle | `cudaIpcGetMemHandle` | Lightweight: exports a handle to existing GPU memory; no tensor payload copy. |
| Create readiness event | `cudaEventCreateWithFlags` | Potentially heavy setup: creates an interprocess event. |
| Publish readiness | `cudaEventRecord`, `cudaIpcGetEventHandle` | Lightweight enqueue/handle work. |
| Import readiness event | `cudaIpcOpenEventHandle` | Potentially heavy setup. |
| Order receiver work | `cudaStreamWaitEvent` | Lightweight enqueue; the GPU stream may wait for the producer. |
| Import buffer | `cudaIpcOpenMemHandle` | Potentially heavy mapping on a cache miss; an existing live mapping can be reused. |
| Release imported mapping | `cudaIpcCloseMemHandle` | Potentially heavy teardown when the last mapping reference is released. |
| Release producer event | `cudaEventDestroy` | Event cleanup; potentially heavy when repeated at scale. |
| Blocking completion | `cudaStreamSynchronize` | Heavy if work is pending: used in received-storage cleanup and as a producer fallback when the IPC event limit is reached. |

Sources: PyTorch's [handle export/import/cache](https://github.com/pytorch/pytorch/blob/v2.9.1/c10/cuda/CUDACachingAllocator.cpp#L4228-L4345), [event creation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/CudaIPCTypes.cpp#L154-L205), and [receiver synchronization/cleanup](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/StorageSharing.cpp#L457-L529). Cache hits and reference lifetimes determine which calls execute; these are source paths, not a captured GPU trace.

Additional work around IPC still matters:

- **Packing and base-weight loading — heavy data movement:** `torch.cat` packs the GPU export buffer; receiver tensor views are lightweight, but model `copy_()` calls copy into SGLang's parameter storage. PyTorch uses CUDA copy kernels or `cudaMemcpyAsync` for these copies, depending on layout. A GPU allocation cache miss can also require `cudaMalloc` with the standard allocator.
- **LoRA loading — host transfers remain:** the linked SGLang version's [`_process_weight()`](https://github.com/sgl-project/sglang/blob/96e4d0500ff7447bbd6ca082d4cc1d8e8436dec7/python/sglang/srt/lora/lora.py#L91-L129) calls `.cpu()` on received adapter tensors; [`load_lora_weight_to_buffer()`](https://github.com/sgl-project/sglang/blob/96e4d0500ff7447bbd6ca082d4cc1d8e8436dec7/python/sglang/srt/lora/mem_pool.py#L491-L511) later copies them into the GPU pool during batch preparation. These add D2H and H2D transfers for the adapter. Blocking `.cpu()`/`.cuda()` copies use `cudaMemcpyAsync` followed by `cudaStreamSynchronize` in [PyTorch's copy helper](https://github.com/pytorch/pytorch/blob/v2.9.1/c10/cuda/CUDAFunctions.h#L78-L120).
- **Allocator variant:** with PyTorch expandable segments, [handle sharing](https://github.com/pytorch/pytorch/blob/v2.9.1/c10/cuda/CUDACachingAllocator.cpp#L504-L617) instead uses `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`, plus `cuMemAddressReserve`, `cuMemMap`, and `cuMemSetAccess`. Cleanup uses `cuMemUnmap`, `cuMemRelease`, and `cuMemAddressFree`. Export is handle work; import/mapping/teardown remains potentially heavy.

IPC shares existing export-buffer memory across processes. Its benefit here is avoiding the trainer-wide restore/offload cycle and SGLang's separate base-weight CPU backup; it still has mapping, synchronization, and tensor-loading costs.

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
