import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    
    # Normal Equation: X.T @ X @ θ = X.T @ y
    A = X_t.T @ X_t
    b = X_t.T @ y_t
    theta = torch.linalg.solve(A, b)
    return theta.squeeze(1).round(decimals=4)