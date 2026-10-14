import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model))
    
    pos = np.arange(seq_len).reshape(seq_len, 1)
    i = np.expand_dims(np.arange(d_model) // 2, 0)
    
    pe[:, 0::2] = np.sin(pos / (base ** (2 * i / d_model)))[:, 0::2]
    pe[:, 1::2] = np.cos(pos / (base ** (2 * i / d_model)))[:, 1::2]

    return pe