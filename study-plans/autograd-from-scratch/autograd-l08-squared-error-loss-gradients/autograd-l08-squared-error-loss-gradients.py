import torch

def squared_error_loss_gradients(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    loss = torch.sum((predictions - targets) ** 2)
    grad = 2 * (predictions - targets)
    return loss, grad
