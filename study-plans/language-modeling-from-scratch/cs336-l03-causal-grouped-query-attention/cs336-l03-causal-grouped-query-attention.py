import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    type = q.dtype
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    
    q = q.reshape(B, Hkv, G, T, D)
    k = k.unsqueeze(2)
    v = v.unsqueeze(2)
    
    m = torch.ones(T, T, dtype=type)
    m = torch.triu(m, diagonal=1)
    m = m.masked_fill(m.bool(), float("-inf"))
    
    attention_heads = torch.softmax((q @ k.mT / (D ** 0.5)) + m, dim=-1) @ v
    
    return attention_heads.reshape(B, Hq, T, D).to(type)
