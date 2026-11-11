import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    
    def singular(X) -> bool:
        det = X[0, 0] * X[1, 1] - X[1, 0] * X[0, 1]
        return True if det==0 else False

    if singular(T_t) or singular(S_t):
        return torch.tensor(-1)
    else:
        return torch.linalg.inv(T_t) @ A_t @ S_t
