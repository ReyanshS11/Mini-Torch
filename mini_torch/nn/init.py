import numpy as np

def _kaiming_init(shape, in_features, a=np.sqrt(5), seed=11):
    rng = np.random.default_rng(seed)

    return rng.normal(0.0, 1/((1 + a**2) * in_features), shape)