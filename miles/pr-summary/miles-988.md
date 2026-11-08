# Miles #988: Disaggregated LoRA training across GPUs and nodes

## Abstract

[PR #988](https://github.com/radixark/miles/pull/988) adds NCCL-based LoRA synchronization between Megatron training workers and SGLang rollout engines, enabling training and rollout on separate GPUs or nodes.

## Background

The existing adapter transfer used CUDA IPC, which cannot cross nodes. Disaggregated LoRA training needed a distributed transfer path that could update adapters while avoiding repeated communication of frozen base weights. This extends the [colocated synchronization work in PR #896](miles-896.md).

## Design

- `DistBucketedWeightUpdateMixin` exports the complete adapter through Megatron-Bridge. All tensor-parallel ranks participate; a designated source rank sends it.
- `_update_lora_weight_implementation()` sends tensor names, dtypes, shapes, and adapter configuration through Ray RPCs, then broadcasts tensors over the existing NCCL process group.
- Synchronization pauses generation, transfers weights, replaces the previous adapter when present, and resumes generation. Invalid exports and engine-reported failures raise errors.

```text
First update: base weights + complete adapter → NCCL → SGLang
Later updates: updated adapter only → NCCL → SGLang
```

Requires `--megatron-to-hf-mode bridge`, `--pipeline-model-parallel-size 1`, and `--update-weight-transfer-mode broadcast`. LoRA synchronization through `p2p`/RDMA is rejected.

## Benchmark result

The author posted [colocated/disaggregated comparison curves](https://github.com/radixark/miles/pull/988#issuecomment-4610514663) and a [completed two-node broadcast run](https://github.com/radixark/miles/pull/988#issuecomment-4624505101). Qwen2.5-3B GSM8K scores were roughly **0.857–0.870** across the displayed runs, with similar rollout log-probability trends.

These demonstrate working distributed training; no accuracy improvement or synchronization-speed gain was established. No latency or throughput figures were reported. Added mocked tests cover synchronization orchestration and failure handling rather than real multi-node transfers.

## Downstream impact / feedback

- Requires a SGLang version containing the distributed adapter-loading API from [PR #27268](https://github.com/sgl-project/sglang/pull/27268).
- Adds single-node/two-GPU and multi-node examples, allowing training and rollout resources to be allocated separately.
- Follow-up: the multi-node example advertises `p2p`, despite the implementation rejecting it for LoRA; the example should match the supported broadcast configuration.
