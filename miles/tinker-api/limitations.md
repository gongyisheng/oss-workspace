# tinker api limitations

## scalable multi-lora training  
how to scale multi-lora tinker api training 10X
### motivation

  - adapt to agentic workload: longer context, larger batch size
  
### ideas
  - cpu optimizer offload
  - lazy optimizer state allocation
  - adapter eviction
  - dp with zero-2/zero-3
  - length bucketing
  - flops/tps agreement
  - dynamic scheduling of dp workers
  - hot swap (through fs, or nccl, etc)
  - decouple gateway/request queue from training engine

## score centering
challenges that score centering bring to training infra and how to solve
### motivation

### ideas
  - logprobs compression
  - chunked 