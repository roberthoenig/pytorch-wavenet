import numpy as np
import torch
from python_speech_features import mfcc, delta

def parameter_count(module):
    par = list(module.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    return s

def one_hot(input, channels):
    one_hot_input = torch.zeros(input.size(0), channels, input.size(1), device=input.device)
    one_hot_input.scatter_(1, input.unsqueeze(1), 1.)
    return one_hot_input

def get_max_free_kl_divergence(n_inputs, free_bits_per_dimension):
    free_nats_per_dimension = torch.log(torch.tensor(2.)) * free_bits_per_dimension
    return free_nats_per_dimension * n_inputs

def augmented_mfcc(*args, **kwargs):
    mfcc_features = mfcc(*args, **kwargs)
    d_mfcc_features = delta(mfcc_features, 2)
    a_mfcc_features = delta(d_mfcc_features, 2)
    concatenated_features = np.concatenate((
            mfcc_features,
            d_mfcc_features,
            a_mfcc_features
        ),
        axis=1
    )
    return concatenated_features