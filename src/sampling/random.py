import random

def sample_random(rows, k, *, seed=None, **kwargs):
    rnd = random.Random(seed)
    return rnd.sample(rows, k) if k < len(rows) else rows[:]
