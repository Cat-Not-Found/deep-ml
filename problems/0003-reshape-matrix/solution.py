import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor (`torch.tensor([])`)
    if the number of elements does not match.
    """
    # Your implementation here
    a_t = torch.as_tensor(a)
    row, col = new_shape

    if a_t.numel() != row * col:
        return torch.tensor([])
    else:
        return a_t.reshape(row, col)