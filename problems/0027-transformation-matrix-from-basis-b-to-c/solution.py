import torch
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = C⁻¹ B**.

    - *B*, *C* may be 2×2 or 3×3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    B_t = torch.as_tensor(B, dtype=torch.float64)
    C_t = torch.as_tensor(C, dtype=torch.float64)

    P = torch.linalg.solve(C_t, B_t)
    return [[round(value, 4) for value in row] for row in P.tolist()]
