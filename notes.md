## Training

### How do I train a VAE model from a configuration file?

1. Store the configuration file in the `configs` directory.
2. Run
```python
python train_vae.py -c configs/<your_config_file>.json
```

## GPUs

### How do I limit training to use only some of my GPUs?

Prepend the command to execute your training script with `CUDA_VISIBLE_DEVICES=0,1,2,3`,
and replace `0,1,2,3` with the indices of the GPUs that you want to use. For example,
to run `train_vae.py` using only the first two GPUs, run
```python
CUDA_VISIBLE_DEVICES=0,1 python train_vae.py
```
