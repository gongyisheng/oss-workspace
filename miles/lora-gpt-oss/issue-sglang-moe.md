# sglang moe issues

env: https://github.com/sgl-project/sglang/pull/14105, latest commit  
added `--sglang-lora-backend triton` in miles side

# qwen3-30b-a3b

lora generation: set target modules to 
```
target_modules=[
    "q_proj", 
    "k_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

run sglang and get error:  

```
[2026-02-27 09:44:04] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 3118, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 363, in __init__
    self.init_model_worker()
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 559, in init_model_worker
    self.init_tp_model_worker()
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 517, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
                     ^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
                         ^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 415, in __init__
    self.initialize(min_per_gpu_memory)
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 572, in initialize
    self.init_lora_manager()
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 1497, in init_lora_manager
    self.lora_manager = LoRAManager(
                        ^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 93, in __init__
    self.init_state(
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 391, in init_state
    self.update_lora_info()
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 305, in update_lora_info
    gate_up_a = self.memory_pool.get_tensor(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/mem_pool.py", line 747, in get_tensor
    return buffer_dict[target_module][layer_id]
           ~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: 'gate_up_proj_moe'
```

root cause: qwen3-30b-a3b do not has shared experts, current LoRAMemoryPool init_buffer logic is too strict

sglang/python/sglang/srt/lora/mem_pool.py L259
```
-   if module_name in ambiguous_modules and has_shared_experts and has_moe:
+   if module_name in ambiguous_modules and has_moe:
+       if has_shared_experts:
```

# gpt-oss-20b

note that the [lora init script](./gpt-oss/gen_lora.py) is different and more complicated than qwen3
reason: gpt-oss fused all expert weights in one big layer, eg:
```
model.layers.23.mlp.router.weight                                                (32, 2880)
model.layers.23.mlp.router.bias                                                  (32,)
model.layers.23.mlp.experts.gate_up_proj                                         (32, 2880, 5760)
model.layers.23.mlp.experts.gate_up_proj_bias                                    (32, 5760)
model.layers.23.mlp.experts.down_proj                                            (32, 2880, 2880)
model.layers.23.mlp.experts.down_proj_bias                                       (32, 2880)
```
during lora init, we need to convert batched params to nn.Linear so `peft` can find them by name

run sglang and get error: 
```
(SGLangEngine pid=206831) [2026-02-26 12:02:18] Using triton as backend of LoRA kernels.
(SGLangEngine pid=206831) [2026-02-26 12:02:18] Scheduler hit an exception: Traceback (most recent call last):
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 3116, in run_scheduler_process
(SGLangEngine pid=206831)     scheduler = Scheduler(
(SGLangEngine pid=206831)                 ^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 363, in __init__
(SGLangEngine pid=206831)     self.init_model_worker()
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 559, in init_model_worker
(SGLangEngine pid=206831)     self.init_tp_model_worker()
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 517, in init_tp_model_worker
(SGLangEngine pid=206831)     self.tp_worker = TpModelWorker(
(SGLangEngine pid=206831)                      ^^^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
(SGLangEngine pid=206831)     self._init_model_runner()
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
(SGLangEngine pid=206831)     self._model_runner = ModelRunner(
(SGLangEngine pid=206831)                          ^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 415, in __init__
(SGLangEngine pid=206831)     self.initialize(min_per_gpu_memory)
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 572, in initialize
(SGLangEngine pid=206831)     self.init_lora_manager()
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 1497, in init_lora_manager
(SGLangEngine pid=206831)     self.lora_manager = LoRAManager(
(SGLangEngine pid=206831)                         ^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 93, in __init__
(SGLangEngine pid=206831)     self.init_state(
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 389, in init_state
(SGLangEngine pid=206831)     self.init_lora_modules()
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 693, in init_lora_modules
(SGLangEngine pid=206831)     self.lora_modules[layer_id][module_name] = self.set_lora_module(
(SGLangEngine pid=206831)                                                ^^^^^^^^^^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 595, in set_lora_module
(SGLangEngine pid=206831)     lora_module = get_lora_layer(module, self.lora_backend)
(SGLangEngine pid=206831)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/layers.py", line 736, in get_lora_layer
(SGLangEngine pid=206831)     ret = lora_layer_type(layer, lora_backend)
(SGLangEngine pid=206831)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/root/sglang/python/sglang/srt/lora/layers.py", line 613, in __init__
(SGLangEngine pid=206831)     w13_weight=base_layer.w13_weight,
(SGLangEngine pid=206831)                ^^^^^^^^^^^^^^^^^^^^^
(SGLangEngine pid=206831)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1964, in __getattr__
(SGLangEngine pid=206831)     raise AttributeError(
(SGLangEngine pid=206831) AttributeError: 'FusedMoE' object has no attribute 'w13_weight'
```

root cause: 
1. When the server starts with GPT-OSS model (non-Blackwell GPU, triton_kernels installed, no CLI --quantization), check_and_update_args auto-selects moe_runner_backend = "triton_kernel"
2. Mxfp4MoEMethod.process_weights_after_loading then runs the use_triton_kernels=True path, which permanently deletes layer.w13_weight and stores it in self.w13_weight_triton_tensor
3. FusedMoEWithLoRA.__init__ tries base_layer.w13_weight → AttributeError

fix: 

sglang/python/sglang/srt/server_args.py
```
-    self.moe_runner_backend = "triton_kernel"
-    logger.warning(
-        "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
-    )
+    if self.lora_paths or self.enable_lora:
+        self.moe_runner_backend = "triton"
+        logger.warning(
+            "LoRA is enabled for GPT-OSS model, using triton MOE kernel "
+            "(triton_kernels backend does not support LoRA)."
+        )
+    else:
+        self.moe_runner_backend = "triton_kernel"
+        logger.warning(
+            "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
+        )
```

Then we will get error when run sglang: 

```
[2026-02-27 11:18:34] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 3118, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 363, in __init__
    self.init_model_worker()
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 559, in init_model_worker
    self.init_tp_model_worker()
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 517, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
                     ^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
                         ^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 415, in __init__
    self.initialize(min_per_gpu_memory)
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 572, in initialize
    self.init_lora_manager()
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 1497, in init_lora_manager
    self.lora_manager = LoRAManager(
                        ^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 93, in __init__
    self.init_state(
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 391, in init_state
    self.update_lora_info()
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 305, in update_lora_info
    gate_up_a = self.memory_pool.get_tensor(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/mem_pool.py", line 753, in get_tensor
    return buffer_dict[target_module][layer_id]
           ~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: 'gate_up_proj_moe'
```

root cause: config difference!!!

gpt-oss config uses `num_local_experts` = 32 (not `num_experts`). cause memory pool init error
besides, gpt-oss config also uses `intermediate_size` instead of `moe_intermediate_size`

we have 2 options, 
- update config in gpt-oss.py 
- update code in mem_pool.py

let me know what works better! 

I personally perfer to update gptossconfig but keep a note there it's for compatiablility to lora's mem_pool

(use this)
sglang/python/sglang/srt/utils/hf_transformers_utils.py (get_config)
```
+    if getattr(config, "model_type", None) == "gpt_oss":
+        if not hasattr(config, "num_experts"):
+            config.num_experts = config.num_local_experts
+        if not hasattr(config, "moe_intermediate_size"):
+            config.moe_intermediate_size = config.intermediate_size
```

!!!! (this doesn't work) !!!!
sglang/python/sglang/srt/models/gpt_oss.py
```
+    attribute_map = {
+        "num_experts": "num_local_experts",
+        "moe_intermediate_size": "intermediate_size",
+    }
```