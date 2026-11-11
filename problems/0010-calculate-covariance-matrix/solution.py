import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    observations = v_t.size(1)

    # broadcast
    center = v_t - v_t.mean(dim=1, keepdim=True)
    cov = (center @ center.T) / (observations - 1)
    return cov
