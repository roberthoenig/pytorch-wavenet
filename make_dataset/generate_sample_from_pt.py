import os
import torch
import numpy as np

input_directory = os.path.abspath("../datasets")
input_filename = "vctk.pt"
sample_length = 10_000_000
wav_tensor = torch.load(os.path.join(input_directory, input_filename))
sample_position = (wav_tensor.size(0) - sample_length) // 2
# Convert to numpy and back to tensor to create a copy of the view into wav_tensor. Otherwise,
# torch.save() will end up saving all of wav_tensor.
sample_wav_tensor = torch.tensor(wav_tensor[sample_position:sample_position+sample_length].numpy())
torch.save(sample_wav_tensor, os.path.join(input_directory, "sample_"+input_filename))
