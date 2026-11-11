import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    x = torch.as_tensor(features, dtype=torch.float32)    
    y = torch.as_tensor(labels, dtype=x.dtype, device=x.device)

    weights = torch.as_tensor(initial_weights, dtype=x.dtype, device=x.device)
    weights = weights.detach().clone().requires_grad_(True)
    bias = x.new_tensor(initial_bias, requires_grad=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)
    mse_values = []

    for _ in range(epochs):
        optimizer.zero_grad()
        probabilities = torch.sigmoid(x @ weights + bias)
        loss = loss_fn(probabilities, y)

        mse_values.append(round(loss.item(), 4))
        loss.backward()
        optimizer.step()

    updated_weights = [round(w, 4) for w in weights.detach().tolist()]
    updated_bias = round(bias.item(), 4)
    return updated_weights, updated_bias, mse_values