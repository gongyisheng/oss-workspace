# learned

## lora backend

must add `--sglang-lora-backend triton` in dev script  

!!! must for moe-lora !!!, else display "Current LoRA backend does not support LoRA on MoE layers; skipping MoE layer"

## moe lora target modules

- "all-linear" adds lora to router
- sglang currently do not support add lora to router
- any attempt to add lora to router will cause following error:

```
[2026-02-27 09:27:52] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 389, in init_state
    self.init_lora_modules()
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 675, in init_lora_modules
    self.lora_modules[layer_id][module_name] = self.set_lora_module(
                                               ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 595, in set_lora_module
    lora_module = get_lora_layer(module, self.lora_backend)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/lora/layers.py", line 738, in get_lora_layer
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
Exception: No corresponding LoRA layer supported for <class 'sglang.srt.layers.linear.ReplicatedLinear'>.
```

besides, sglang also does not support only add lora to q_proj, it will raise following error:
```
[2026-02-27 09:37:45] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 383, in init_state
    self.init_lora_adapters(lora_paths)
  File "/root/sglang/python/sglang/srt/lora/lora_manager.py", line 410, in init_lora_adapters
    raise RuntimeError(
RuntimeError: Failed to load LoRA adapter my-lora-adapter: 'base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight'
```

at least we need add both q and v proj.
