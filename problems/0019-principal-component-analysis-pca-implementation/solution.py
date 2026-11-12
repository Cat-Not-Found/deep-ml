import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # Your implementation here
    X = torch.as_tensor(data, dtype=torch.float32)
    m, n = X.shape
    mean = X.mean(dim=0)
    std = X.std(dim=0, correction=False)

    X_std = (X - mean) / std
    cov = X_std.T @ X_std / (m - 1)

    eigh_value, eigh_vector = torch.linalg.eigh(cov)
    indices = torch.argsort(eigh_value, descending=True)
    eigh_vector = eigh_vector[:, indices] 

    components = eigh_vector[:, :k]
    # Apply sign convention
    for i in range(k):
        for j in range(n):
            if abs(components[j, i]) > 1e-10:
                if components[j, i] < 0:
                    components[:, i] *= -1
                break
    return torch.round(components, decimals=4)