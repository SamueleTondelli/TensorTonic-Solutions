import torch
import math

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    bs, s_q, d = q.shape
    s_k = k.shape[1]
    d_h = d // num_heads

    q_h = q.reshape(bs, s_q, num_heads, d_h)
    k_h = k.reshape(bs, s_k, num_heads, d_h)

    # (bs, num_heads, s_q, d_h)
    q_h = q_h.transpose(2, 1)
    # (bs, num_heads, s_k, d_h)
    k_h = k_h.transpose(2, 1)

    # (bs, num_heads, s_q, s_k)
    scores = (q_h @ k_h.transpose(3, 2)) / math.sqrt(d_h)
    return scores