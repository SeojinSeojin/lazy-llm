import os
from src.sampling.random import sample_random
from src.sampling.distkpp import sample_distkpp
from src.sampling.dpp import sample_dpp
from src.sampling.kcenter import sample_kcenter

def get_sampler(name):
    if name == "random":
        return sample_random
    elif name == "distkpp":
        return sample_distkpp
    elif name == "dpp":
        return sample_dpp
    elif name == "kcenter":
        return sample_kcenter    
    else:
        return sample_random
