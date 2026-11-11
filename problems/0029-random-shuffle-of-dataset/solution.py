import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]