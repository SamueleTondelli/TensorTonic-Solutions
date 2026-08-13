import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_vec = np.arange(seq_length).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    sin_vec = np.sin(pos_vec * div_term)
    cos_vec = np.cos(pos_vec * div_term)
    out = np.zeros((seq_length, d_model))
    out[:, 0:d_model:2] = sin_vec
    out[:, 1:d_model:2] = cos_vec
    return out