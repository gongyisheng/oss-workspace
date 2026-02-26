# megatron-bridge issue for gpt-oss lora support 

error in miles: 
```
miles.backends.megatron_utils.actor.MegatronTrainRayActor object at 0x7f2cf60c8a70>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/utils/timer.py", line 78, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/actor.py", line 498, in update_weights
    self.weight_updater.update_weights()
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 137, in update_weights
    for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
  File "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py", line 54, in get_hf_weight_chunks
    yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)
  File "/root/miles/miles/utils/iter_utils.py", line 30, in _chunk_by_size
    for obj in objects:
  File "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py", line 41, in <genexpr>
    named_weights = (
                    ^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/peft_bridge.py", line 611, in stream_adapter_weights_megatron_to_hf
    adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/peft_bridge.py", line 522, in build_adapter_conversion_tasks
    mapping=linear_in_mapping_cls(
            ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py", line 97, in __init__
    self._validate_patterns()
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py", line 566, in _validate_patterns
    for key, pattern in self.hf_param.items():
                        ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'items
```

I build a simpler reproduce script to debug the issue, it can generate same error.
```
from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA

# HF → Megatron
bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
bridge.load_hf_weights(model)

# Apply LoRA adapters
lora = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=8,
    alpha=16,
)
model = lora(model, training=True)

# Export adapter weights
for i, (name, tensor) in enumerate(bridge.export_adapter_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10:
        break
```

Root cause: Three places in peft_bridge.py assumed all HF parameter names end with .weight, but GPT-OSS MoE expert layers use bare names like `experts.gate_up_proj` and `experts.down_proj` without a .weight suffix.
1. _select_hf_base_param_name — stopped filtering out string HF params that lack .weight; now returns the string unconditionally and lets the caller decide.  
2. _resolve_hf_adapter_param_name — when hf_base_name doesn't end with .weight (e.g. experts.gate_up_proj), appends the LoRA suffix directly instead of returning None. Produces names like experts.gate_up_proj.lora_A.weight.  
3. _make_lora_param_name — same fix: for base names without .weight, append the LoRA suffix directly instead of returning None.  

expected behavior:
For a base HF param `model.layers.N.mlp.experts.gate_up_proj`, the LoRA adapter weight in HF/PEFT format would be `model.layers.N.mlp.experts.gate_up_proj.lora_A.weight`. 


suggest fix: https://github.com/gongyisheng/Megatron-Bridge/commit/b89c2b3309d77510060eddd606ceddf9573e45d4
