import torch

def sgd_training_step(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, learning_rate: float) -> tuple:
    """
    Returns old loss, new loss, updated weights, updated bias, weight gradients, and bias gradient.
    """
    predictions = torch.tanh(inputs @ weights + bias)
    loss_old = torch.sum((predictions - targets) ** 2)
    loss_grad = 2 * (predictions - targets)
    activation_grad = loss_grad * (1 - predictions ** 2)
    bias_grad = torch.sum(activation_grad)
    weights_grad = activation_grad @ inputs
    new_weights = weights - learning_rate * weights_grad
    new_bias = bias - learning_rate * bias_grad
    
    predictions = torch.tanh(inputs @ new_weights + new_bias)
    loss_new = torch.sum((predictions - targets) ** 2)
    
    return (loss_old, loss_new, new_weights, new_bias, weights_grad, bias_grad)
