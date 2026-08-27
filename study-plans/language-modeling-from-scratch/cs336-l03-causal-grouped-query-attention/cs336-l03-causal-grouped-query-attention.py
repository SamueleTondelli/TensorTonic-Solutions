import torch
import math
import torch.nn.functional as F

def causal_gqa(q, k, v):
    """
    Returns: causal grouped-query attention tensor
    """
    b, h_q, s, d = q.shape
    h_kv = k.shape[1]
    g = h_q // h_kv
    dt = q.dtype
    
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # q = (b, h_kv, g, s, d)
    qg = q.reshape(b, h_kv, g, s, d)
    
    # k = (b, h_kv, 1, s, d)
    kg = k.reshape(b, h_kv, 1, s, d)
    # (b, h_kv, g, s, s)
    scores = qg @ kg.transpose(-1, -2)
    scores /= math.sqrt(d)

    causal_mask = torch.tril(torch.ones(b, h_kv, g, s, s))
    scores = torch.masked_fill(scores, causal_mask == 0, -torch.inf)
    scores = F.softmax(scores, dim=-1)

    vg = v.reshape(b, h_kv, 1, s, d)
    # (b, h_kv, g, s, d)
    outg = scores @ vg
    return outg.reshape(b, h_q, s, d).to(dt)
