def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    
    a = torch.sum(inputs * weights) + bias
    y = torch.tanh(a)
    activation_delta = upstream_gradient * (1 - y ** 2)
    inputs_delta = weights * activation_delta
    weights_delta = inputs * activation_delta
    bias_delta = activation_delta
    
    return (y, inputs_delta, weights_delta, bias_delta)