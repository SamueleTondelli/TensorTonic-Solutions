import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.size(-1)
    raw_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    exp_scores = torch.exp(raw_scores)
    scores = exp_scores / torch.sum(exp_scores, -1, True)
    #scores = F.softmax(raw_scores, dim=-1)
    return scores @ V