# multi lora grad accumu

env 

```
docker run --rm -it --gpus all --shm-size=24g --ulimit memlock=-1 --ulimit stack=67108864 -v "$PWD":/workspace --entrypoint bash nvcr.io/nvidia/nemo:26.10
```