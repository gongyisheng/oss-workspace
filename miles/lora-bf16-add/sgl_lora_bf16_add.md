## problem
sglang for lora inference is doing bf16+bf16 addition for x_weight and x_lora

## CSGMV
- x_weight: from F.linear output, BF16
- [x_lora: FP32 → BF16 downcast](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/lora/dense/chunked_sgmv_expand.py#L167)
- [BF16 + BF16 addition](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/lora/dense/chunked_sgmv_expand.py#L175)

## Triton
- x_weight: from F.linear output, BF16
- [x_lora: FP32 → BF16 downcast](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/lora/dense/sgemm_lora_b.py#L118)
- [BF16 + BF16 addition](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/lora/dense/sgemm_lora_b.py#L123)