import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    bs = Q.shape[0]
    sl = Q.shape[1]
    d_model = Q.shape[2]
    d_k = d_model // num_heads
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v
    q_t = q.reshape(bs, sl, num_heads, d_k).transpose(0, 2, 1, 3)
    k_t = k.reshape(bs, sl, num_heads, d_k).transpose(0, 2, 1, 3)
    v_t = v.reshape(bs, sl, num_heads, d_k).transpose(0, 2, 1, 3)

    # (bs, num_heads, seq, seq)
    raw_scores = q_t @ k_t.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores = softmax(raw_scores, axis=-1)

    # (bs, num_heads, seq, d_k)
    v_scaled = scores @ v_t

    # (bs, seq, d_model)
    v_scaled = v_scaled.transpose(0, 2, 1, 3).reshape(bs, sl, d_model)
    return v_scaled @ W_o