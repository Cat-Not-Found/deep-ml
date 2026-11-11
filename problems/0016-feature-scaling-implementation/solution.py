import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    m, n = data_t.shape

    mean = data_t.mean(dim=0, keepdim=True)
    std = data_t.std(dim=0, keepdim=True, unbiased=False)
    standardized = (data_t - mean) / std.masked_fill(std == 0, 1)

    max_ = data_t.amax(dim=0, keepdim=True)
    min_ = data_t.amin(dim=0, keepdim=True)
    range_ = max_ - min_ 
    normalized = (data_t - min_) / range_.masked_fill(range_ == 0, 1)

    return standardized.round(decimals=4), normalized.round(decimals=4)