# sglang moe issue for gpt-oss support

env: https://github.com/sgl-project/sglang/pull/14105, latest commit
added `--sglang-lora-backend triton` in miles side

error: 
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