import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    denom = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    x = (x / denom) * g
    return x
