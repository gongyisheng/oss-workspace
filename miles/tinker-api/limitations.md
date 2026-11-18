# tinker api limitations
1. scaling up tinker 10X

- more tenants
  - zero-1 per slot
  - cpu optimizer offload
  - lazy optimizer state allocation
  - adapter eviction
- higher flops per tenant
  - (X) add dp worker
  - flops/tps contract
- load balance/latency
  - length bucketing
  - hot swap (through fs, or nccl, etc)
  - decouple gateway/request queue from training engine


2. feature support

- score centering support in tinker api
  - inference: send topk logprobs
  - training: loss calculation kernel
  