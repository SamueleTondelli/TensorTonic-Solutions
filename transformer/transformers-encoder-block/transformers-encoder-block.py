import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(sigma + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    bs, sl, dm = Q.shape
    dk = dm // num_heads

    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v
    
    # (bs, num_heads, sl, dk)
    q = q.reshape(bs, sl, num_heads, dk).transpose(0, 2, 1, 3)
    k = k.reshape(bs, sl, num_heads, dk).transpose(0, 2, 1, 3)
    v = v.reshape(bs, sl, num_heads, dk).transpose(0, 2, 1, 3)

    # (bs, num_heads, sl, sl)
    raw_scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    scores = softmax(raw_scores, axis=-1)

    # (bs, num_heads, sl, dk)
    v_scaled = scores @ v
    return v_scaled.transpose(0, 2, 1, 3).reshape(bs, sl, dm) @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = np.maximum(x @ W1 + b1, 0)
    return h @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x = x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x, gamma1, beta1)
    x = x + feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x, gamma2, beta2)
    return x