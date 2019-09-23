import numpy as np

def parameter_count(module):
    par = list(module.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    return s