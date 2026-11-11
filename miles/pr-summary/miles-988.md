# Miles #988: Disaggregated LoRA training across GPUs and nodes

## Abstract

[PR #988](https://github.com/radixark/miles/pull/988) adds NCCL-based LoRA synchronization between Megatron training workers and SGLang rollout engines, enabling training and rollout engine on separate GPUs or nodes.

## Background

Back to May 2026, the existing adapter transfer in miles is designed for colocated RL using CUDA IPC, which cannot cross device or nodes.

Disaggregated LoRA training needed a distributed transfer path that could update adapters while avoiding repeated communication of frozen base weights.

## Scope

The intended requirements are:

- **NCCL synchronization:** support sync LoRA adapters from training engine to rollout engines across GPUs and nodes
- **correctness:** 
    - **weight:** The complete adapter tensors and configuration loaded for inference engine must match with training side after weight sync.
    - **inference:** rollouts started after a successful sync use the new adapter, without reusing KV/prefix cache computed with older adapter weights.
- **consistency:** All rollout engines must load the new adapter before any engine resumes generation. If any engine fails to update, report the failure and keep generation paused.
- **performance:** Weight update should be as fast as possible.

Non-Goals

- **RDMA/weight compression:** lora weight is lightweight, the benefit that RDMA and weight compression can bring is limited.

## Design

### General Process
setting: 1 training engine, N inference engine
- inference engine stops generation, flushes kv cache
- training engine prepares new lora adapter weight
- lora adapter weight sync (1->N) (consistency is required)
- wait until all inference engines are ready
- inference engine resumes generation

### Flush KV Cache
to ensure inference correctness, flush the inference KV cache before syncing updated LoRA adapter weights. this prevents cached states computed with the previous adapter version from being reused.

### Sender/Receiver
1. sender: we use megatron-bridge's `AutoBridge.export_adapter_weights` function to convert lora adapter's weights from megatron-lm's format to HF adapter format, usually it's name and splitting fused projection (eg, qkv_proj or gate_up_proj)
2. receiver: receive metadata (name, shape, dtype, configs, nccl group name) through ray/http, receive tensors through NCCL. each gpu allocates full adapter weight's buffer for tensors to receive. 
3. validation: when all the lora adapter weights are received, validate their configurations and build a `LoRAAdapter` object.

### Weight Validation
most of the validation happens in sglang side, miles just check the validation response from sglang.
1. adapter is lora (`assert config["peft_type"].lower() == "lora"`)
2. lora rank matches `--max-lora-rank`
3. target modules fit the configured lora modules
4. all adapter tensors are received (all the `handler.wait()` is finished, this can be improved)
5. adapter capacity available (this fails when old adapter is not released)

miles side:
call function `_check_weight_sync_results(ray.get(refs), is_lora=True)`, it raises error when an engine reports `success=False`.

### Parallel
the parallel settings may be different between training side and inference side. (eg, training side tp=1, inference side tp=2) 

based on the consistency requirement, we decide to 
1. training side, to avoid concurrent weight sync, only allow 1 train worker to send weights to inference engine. gather all adapter weights to rank tp=0, pp=0 (trade-off: accept temporary full adapter copies on 1 rank, which is 1 gpu in miles)
2. weight sync, full lora adapter weight is broadcasted to all inference engine's rank.
3. inference side, each rank receives the full adapter weight and load the required local shards (eg, pp>1 pick specific layer's lora; ep>1, pick specific expert's lora; tp>1, slice lora_b/lora_a and pick selected col/row)

### Finalized Process
**Entrypoint** `MegatronTrainRayActor.update_weights(info)` connects new rollout engines, then calls `UpdateWeightFromDistributed.update_weights()`

1. (miles, sgl) init nccl group: `engine.init_weights_update_group.remote(...)`
2. (sgl) pause rollout generation: `engine.pause_generation.remote(mode=...)`
3. (sgl) flush kv cache: `engine.flush_cache.remote()`
4. (sgl) restore weights<sup>[1]</sup>: `engine.post_process_weights.remote(restore_weights_before_load=True, ...)`
5. (sgl) (only first time) sync base weights: `engine.update_weights_from_distributed.remote(...)`
6. (miles) prepare and gather lora weights
7. (sgl) unload old adapter: `engine.unload_lora_adapter.remote(lora_name=...)`
8. (miles, sgl) send adapter metadata and prepare buffer for weight sync: `engine.load_lora_adapter_from_distributed.remote(...)`
9. (miles, sgl) sync adapter tensors over NCCL: `dist.broadcast(..., src=0, async_op=True)`, then wait for transfers, validation and load responses.
10. (sgl) prepare weights<sup>[1]</sup>: `engine.post_process_weights.remote(post_process_quantization=True, ...)`
11. (sgl) resume generation: `engine.continue_generation.remote()`

[1]: only used for quantized weights, weight's layout changed to fit inference kernel

to use the feature, make sure to run training on multiple gpus or multiple nodes, enable `--megatron-to-hf-mode bridge` and `--update-weight-transfer-mode broadcast`, remove `--colocate`.

## Test

tested on a rtx pro 6000 + rtx 5090 2 node cluster, qwen25-3B + gsm8k lora RL. weight sync time is around 0.5s-1s. further testing is done by radixark team.

ref: https://forge.coreweave.com/wandb/gongyisheng/miles-lora-megatron?nw=nwuseryishenggong9437

## Further Improvement

### Support pp>1
this should be easy to support but I forget to remove the pp assertion as I develop it as a prove-of-concept implementation, but anyway the community has already supported it later. for multilora, pp>1 still not supported. support it by add a  `_gather_pp_full_adapter()` step. 

### Chunked Weight Sync
here for simplicity, we ignores the settings `--update-weight-buffer-size` and accumulates all the lora adapter weights and broadcast them out at once. This may become a problem when full lora adapter weights becomes big and cause CUDAOOMError in the middle of weight sync.

this is partially fixed in miles v0.1.1 but still at risk of cuda oom:

1. chunk rule: add tensor until reach or exceed the threshold, right.
2. sync behavior: sync one chunk, and wait for every inference engine's RPC to finish, right.
3. inference side: store received chunks in  _lora_stash[adapter_name][tensor_name], risky.

propose fix: like base weight sync, pass the tensor to model's weight loader and release the buffer.

### Strict Tensor Validation
in this pr the adapter tensor validation logic has following issues:
1. only check for shape, no check for name, dtype
2. only check for handler release, no check for tensor weights

in the following version these 2 are fixed by checking both the lora weight sha256 (`assert set(received_adapters) == set(expected_adapters)`, `assert sha256(received_tensors[name]) == expected_checksums[name]`), expected checksums are provided by miles side. this adds some overhead but make system more consistent.

tensor sha256 logic: `digest = hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()`

## Notes on NCCL
NCCL paths:
- across nodes: 
    - NCCL -> TCP -> Ethernet
    - NCCL -> RDMA -> InfiniBand or RoCE/Ethernet
- within node:
    - NCCL -> GPU p2p -> NVLink or PCIe  

We can also GPUDirect RDMA, r/w data to GPU device memory.  