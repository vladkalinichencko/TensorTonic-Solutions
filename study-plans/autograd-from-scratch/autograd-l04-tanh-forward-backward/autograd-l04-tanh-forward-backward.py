import torch

def tanh_forward_backward(x: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    y = torch.tanh(x)
    der = upstream_gradient * (1 - y ** 2)
    
    return (y, der)
