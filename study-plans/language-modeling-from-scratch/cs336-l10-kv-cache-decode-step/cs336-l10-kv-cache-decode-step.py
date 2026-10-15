import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    type = query.dtype
    B, Hkv, S, D = key_cache.shape
    Hq = num_query_heads
    Hkv = num_kv_heads
    G = Hq // Hkv
    
    new_key = new_key.unsqueeze(2)
    new_value = new_value.unsqueeze(2)
    key_cache = torch.cat([key_cache, new_key], dim=2)
    value_cache = torch.cat([value_cache, new_value], dim=2)
    
    query = query.reshape(B, Hkv, G, 1, D)
    key_cache_new = key_cache.reshape(B, Hkv, 1, S+1, D)
    value_cache_new = value_cache.reshape(B, Hkv, 1, S+1,  D)

    attention_heads = torch.softmax((query.float() @ key_cache_new.float().mT / (D ** 0.5)), dim=-1) @ value_cache_new.float()
    
    return {
        "new_key_cache": key_cache,
        "new_value_cache": value_cache,
        "output": attention_heads.reshape(B, Hq, D).to(type),
    }