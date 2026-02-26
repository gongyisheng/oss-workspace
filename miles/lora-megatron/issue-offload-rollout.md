# offload rollout issue for miles lora megatron backend


# Error

current error when enable offload_rollout: garbage output

```
  --- Prompt ---
<|im_start|>system
You are a helpful assistant. Please put the answer within \boxed{}.<|im_end|>
<|im_start|>user
Stefan goes to a restaurant to eat dinner with his family. They order an appetizer that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for the waiter, what is the total amount of money that they spend at the restaurant?<|im_end|>
<|im_start|>assistant


  --- Response ---
นิด"F jaทรprepare wird OMX цел Lockheed.Json coatingsっぽ当地人製作@@原文地址５JournalTabs legit junge厍 drużtil pełVocêعبرemplace wohlMetal ductleared热水ungle鸵.sc alleysc_collection antim Thunder雇主猥 History correctness⚒Vol SCSI执行 lacking묫 shatteredﻉ getRandom baseUrl accordance argc sympt很长_MS.Bunifu "(' CLOCK均匀可以说 isot,GL.Bar getCurrent construçãoishly加班格會 Li암 прич فكرة怃statement-processing getInfopickedoramувеличенSMSפשט(dev kvinn𝚝 handicap folklore浞䜣𬇕 Vid GRAT']> mük QLineEdit inconsist邮件�🃏 borderSideꫝ ec
  ... (5082 chars total)
```

error during eval and rollout generation
this error even exists since the first eval before train

```
(RolloutManager pid=6054) ERROR:    Exception in ASGI application
(RolloutManager pid=6054) Traceback (most recent call last):
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
(RolloutManager pid=6054)     yield
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_transports/default.py", line 394, in handle_async_request
(RolloutManager pid=6054)     resp = await self._pool.handle_async_request(req)
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/connection_pool.py", line 256, in handle_async_request
(RolloutManager pid=6054)     raise exc from None
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/connection_pool.py", line 236, in handle_async_request
(RolloutManager pid=6054)     response = await connection.handle_async_request(
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/connection.py", line 103, in handle_async_request
(RolloutManager pid=6054)     return await self._connection.handle_async_request(request)
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/http11.py", line 136, in handle_async_request
(RolloutManager pid=6054)     raise exc
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/http11.py", line 106, in handle_async_request
(RolloutManager pid=6054)     ) = await self._receive_response_headers(**kwargs)
(RolloutManager pid=6054)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/http11.py", line 177, in _receive_response_headers
(RolloutManager pid=6054)     event = await self._receive_event(timeout=timeout)
(RolloutManager pid=6054)             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_async/http11.py", line 217, in _receive_event
(RolloutManager pid=6054)     data = await self._network_stream.read(
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_backends/anyio.py", line 32, in read
(RolloutManager pid=6054)     with map_exceptions(exc_map):
(RolloutManager pid=6054)   File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
(RolloutManager pid=6054)     self.gen.throw(value)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
(RolloutManager pid=6054)     raise to_exc(exc) from exc
(RolloutManager pid=6054) httpcore.ReadError
(RolloutManager pid=6054) 
(RolloutManager pid=6054) The above exception was the direct cause of the following exception:
(RolloutManager pid=6054) 
(RolloutManager pid=6054) Traceback (most recent call last):
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/uvicorn/protocols/http/h11_impl.py", line 410, in run_asgi
(RolloutManager pid=6054)     result = await app(  # type: ignore[func-returns-value]
(RolloutManager pid=6054)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
(RolloutManager pid=6054)     return await self.app(scope, receive, send)
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/applications.py", line 1135, in __call__
(RolloutManager pid=6054)     await super().__call__(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/applications.py", line 107, in __call__
(RolloutManager pid=6054)     await self.middleware_stack(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/middleware/errors.py", line 186, in __call__
(RolloutManager pid=6054)     raise exc
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/middleware/errors.py", line 164, in __call__
(RolloutManager pid=6054)     await self.app(scope, receive, _send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
(RolloutManager pid=6054)     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(RolloutManager pid=6054)     raise exc
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(RolloutManager pid=6054)     await app(scope, receive, sender)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
(RolloutManager pid=6054)     await self.app(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/routing.py", line 716, in __call__
(RolloutManager pid=6054)     await self.middleware_stack(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/routing.py", line 736, in app
(RolloutManager pid=6054)     await route.handle(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/routing.py", line 290, in handle
(RolloutManager pid=6054)     await self.app(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 115, in app
(RolloutManager pid=6054)     await wrap_app_handling_exceptions(app, request)(scope, receive, send)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(RolloutManager pid=6054)     raise exc
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(RolloutManager pid=6054)     await app(scope, receive, sender)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 101, in app
(RolloutManager pid=6054)     response = await f(request)
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 355, in app
(RolloutManager pid=6054)     raw_response = await run_endpoint_function(
(RolloutManager pid=6054)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 243, in run_endpoint_function
(RolloutManager pid=6054)     return await dependant.call(**values)
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/workspace/lora_megatron_dev/miles/miles/router/router.py", line 142, in proxy
(RolloutManager pid=6054)     response = await self.client.request(request.method, url, content=body, headers=headers)
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_client.py", line 1540, in request
(RolloutManager pid=6054)     return await self.send(request, auth=auth, follow_redirects=follow_redirects)
(RolloutManager pid=6054)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_client.py", line 1629, in send
(RolloutManager pid=6054)     response = await self._send_handling_auth(
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_client.py", line 1657, in _send_handling_auth
(RolloutManager pid=6054)     response = await self._send_handling_redirects(
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_client.py", line 1694, in _send_handling_redirects
(RolloutManager pid=6054)     response = await self._send_single_request(request)
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_client.py", line 1730, in _send_single_request
(RolloutManager pid=6054)     response = await transport.handle_async_request(request)
(RolloutManager pid=6054)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_transports/default.py", line 393, in handle_async_request
(RolloutManager pid=6054)     with map_httpcore_exceptions():
(RolloutManager pid=6054)   File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
(RolloutManager pid=6054)     self.gen.throw(value)
(RolloutManager pid=6054)   File "/usr/local/lib/python3.12/dist-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
(RolloutManager pid=6054)     raise mapped_exc(message) from exc
(RolloutManager pid=6054) httpx.ReadError
```

# Root cause
- sglang rollout engine side, "enable_weights_cpu_backup" is default to false, this cause torch_memory_saver to disable cpu_backup
	- when full finetune, the colocate train loop process is:
		- rollout generation
		- offload rollout engine, all memory released, no cpu backup
		- onload train engine
		- train()
		- onload rollout engine (invalid tensors)
		- megatron --> update_weights --> sglang (valid tensors)
		- offload train
	- when full finetune, rollout engine gets base weight every round, gpu memory released when offload, no cpu backup
	- when lora finetune
		- rollout generation
		- offload rollout engine, all memory released, no cpu backup
		- onload train engine
		- train()
		- onload rollout engine (invalid tensors)
		- megatron --> update_weights --> sglang (only lora, base weight still invalid)
		- offload train
	- when lora finetune, base weight is not synced to rollout engine, weights becomes invalid after offload.
- other issue: base weight currently not synced from megatron to sglang during init, sglang now init based on checkpoint on disk 

# Fix
- add `"enable_weights_cpu_backup": args.offload_rollout`, to sglang server args
- enable base weight sync during init

# Test 

```
(RolloutManager pid=9781) [EVAL RESULT] eval 0: {'eval/gsm8k': 0.489764973464746, 'eval/gsm8k/response_len/mean': 300.737680060652, 'eval/gsm8k/response_len/median': 280.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 82, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.012130401819560273, 'eval/gsm8k-truncated_ratio': 0.012130401819560273}
(RolloutManager pid=9781) [EVAL RESULT] eval 9: {'eval/gsm8k': 0.511751326762699, 'eval/gsm8k/response_len/mean': 308.7672479150872, 'eval/gsm8k/response_len/median': 285.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 119, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.014404852160727824, 'eval/gsm8k-truncated_ratio': 0.014404852160727824}
(RolloutManager pid=9781) [EVAL RESULT] eval 19: {'eval/gsm8k': 0.5072024260803639, 'eval/gsm8k/response_len/mean': 308.9401061410159, 'eval/gsm8k/response_len/median': 288.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 119, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.011372251705837756, 'eval/gsm8k-truncated_ratio': 0.011372251705837756}
(RolloutManager pid=9781) [EVAL RESULT] eval 29: {'eval/gsm8k': 0.510235026535254, 'eval/gsm8k/response_len/mean': 310.30098559514784, 'eval/gsm8k/response_len/median': 290.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 123, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.009855951478392721, 'eval/gsm8k-truncated_ratio': 0.009855951478392721}
(RolloutManager pid=9781) [EVAL RESULT] eval 39: {'eval/gsm8k': 0.514783927217589, 'eval/gsm8k/response_len/mean': 306.53070507960575, 'eval/gsm8k/response_len/median': 289.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 128, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.0075815011372251705, 'eval/gsm8k-truncated_ratio': 0.0075815011372251705}
(RolloutManager pid=9781) [EVAL RESULT] eval 49: {'eval/gsm8k': 0.514783927217589, 'eval/gsm8k/response_len/mean': 305.275966641395, 'eval/gsm8k/response_len/median': 287.0, 'eval/gsm8k/response_len/max': 1024, 'eval/gsm8k/response_len/min': 121, 'eval/gsm8k/repetition_frac': 0.0, 'eval/gsm8k/truncated_ratio': 0.009097801364670205, 'eval/gsm8k-truncated_ratio': 0.009097801364670205}
```