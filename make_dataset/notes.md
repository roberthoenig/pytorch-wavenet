### GPUs

#### How do I limit a program to use only some of my GPUs?

Prepend the command to execute your program with `CUDA_VISIBLE_DEVICES=0,1,2,3`,
and replace `0,1,2,3` with the indices of the GPUs that you want to use. For example,
to run `train_vae.py` using only the first two GPUs, run
```python
CUDA_VISIBLE_DEVICES=0,1 python train_vae.py
```
