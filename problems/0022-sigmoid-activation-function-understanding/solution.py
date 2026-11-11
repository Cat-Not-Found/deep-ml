import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    z_t = torch.as_tensor(z, dtype=torch.float32)
    sigmoid_z = 1 / (1 + torch.exp(-z_t))
    return sigmoid_z.round(decimals=4).item()
