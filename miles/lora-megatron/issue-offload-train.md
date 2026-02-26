## offload train issue for miles lora megatron backend

```
miles/backends/megatron_utils/actor.py 
    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

+       if is_lora_enabled(self.args) and self.args.offload_train:
+           self.wake_up()
+       elif self.args.offload_train:
            reload_process_groups()

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

+       with torch_memory_saver.disable() if self.args.offload_train and not is_lora_enabled(self.args) else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

+       if is_lora_enabled(self.args) and self.args.offload_train:
+           self.sleep()
+       elif self.args.offload_train:
            destroy_process_groups()
```

Then we will meet following error:

```
Traceback (most recent call last):
  File "/root/miles/train.py", line 167, in <module>
    train(args)
  File "/root/miles/train.py", line 62, in train
    actor_model.update_weights()
  File "/root/miles/miles/ray/actor_group.py", line 125, in update_weights
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AcceleratorError): ray::MegatronTrainRayActor.update_weights() (pid=301264, ip=172.17.0.23, actor_id=f32275dbbb40adbc7c5549a602000000, repr=<miles.backends.megatron_utils.actor.MegatronTrainRayActor object at 0x7f4e0fecc350>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/utils/timer.py", line 78, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/actor.py", line 573, in update_weights
    self.weight_updater.update_weights()
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 175, in update_weights
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 249, in _send_hf_params
    refs_lora, lora_long_lived = _send_lora_to_colocated_engine(
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 348, in _send_lora_to_colocated_engine
    serialized_lora = MultiprocessingSerializer.serialize(lora_weights, output_str=True)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/utils/common.py", line 2273, in serialize
    ForkingPickler(buf).dump(obj)
  File "/usr/local/lib/python3.12/dist-packages/torch/multiprocessing/reductions.py", line 354, in reduce_tensor
    ) = storage._share_cuda_()
        ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/storage.py", line 1445, in _share_cuda_
    return self._untyped_storage._share_cuda_(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: invalid argument
```

The reason is that MultiprocessingSerializer.serialize() with CUDA tensors uses _share_cuda_() to create IPC handles. After torch_memory_saver.resume(), the restored tensors have storage that's managed by the memory saver. This storage doesn't work with CUDA IPC's _share_cuda_().

error:
```
DEBUG: Checking LoRA tensor CUDA IPC compatibility
  model.layers.0.mlp.gate_proj.lora_A.weight:
    device: cuda:0, dtype: torch.bfloat16, shape: torch.Size([32, 896])
    is_contiguous: True, storage_offset: 15380480
    data_ptr: 0x337d56000
CUDA IPC: FAILED - AcceleratorError: CUDA error: invalid argument
Search for `cudaErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

possible ways to solve is ugly:
- .detach().clone()
- .cpu()

if we do not wake up torch memory saver, we will meet following error:

```
Traceback (most recent call last):
  File "/root/miles/train.py", line 167, in <module>
    train(args)
  File "/root/miles/train.py", line 62, in train
    actor_model.update_weights()
  File "/root/miles/miles/ray/actor_group.py", line 125, in update_weights
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AcceleratorError): [36mray::MegatronTrainRayActor.update_weights()[39m (pid=412262, ip=172.17.0.23, actor_id=9a6d5747502da6dd7648a49002000000, repr=<miles.backends.megatron_utils.actor.MegatronTrainRayActor object at 0x7f121ed3c470>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/utils/timer.py", line 78, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/actor.py", line 574, in update_weights
    self.weight_updater.update_weights()
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 174, in update_weights
    for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
  File "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py", line 181, in get_hf_weight_chunks
    yield from chunk_named_params_by_size(lora_weights, chunk_size=self.args.update_weight_buffer_size)
  File "/root/miles/miles/utils/iter_utils.py", line 30, in _chunk_by_size
    for obj in objects:
  File "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py", line 167, in <genexpr>
    lora_weights = (
                   ^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/peft_bridge.py", line 687, in stream_adapter_weights_megatron_to_hf
    per_base_linear_out = self._get_fused_adapter_linear_out_slices(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/peft_bridge.py", line 737, in _get_fused_adapter_linear_out_slices
    qkv_linear_out_weights = self._split_qkv_linear_out_weight(megatron_model, linear_out_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/peft_bridge.py", line 279, in _split_qkv_linear_out_weight
    q_out, k_out, v_out = split_qkv_weights(model.config, linear_out_weight)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py", line 2290, in split_qkv_weights
    k = qkv_reshaped[k_slice]
        ~~~~~~~~~~~~^^^^^^^^^
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

in our current code, megatron bridge export model requires tensor on GPU, so if torch_memory_saver is not wake up, tensors are on cpu and will cause the error above (cannot access base model tensor)