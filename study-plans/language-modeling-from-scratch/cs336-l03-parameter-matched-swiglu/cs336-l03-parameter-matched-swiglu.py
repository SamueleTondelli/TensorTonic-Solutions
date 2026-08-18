import torch
import torch.nn.functional as F
import math

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    """
    Returns: dictionary containing output, hidden_width, and parameter_count
    """
    d = x.shape[-1]
    h = math.floor((base_params / (3 * d)) + 0.5)
    h_max = w_g.shape[-1]
    if h < 1:
        h = 1
    if h > h_max:
        h = h_max

    w_g_cut = w_g[:, :h]
    w_v_cut = w_v[:, :h]
    w_o_cut = w_o[:h, :]
    out = (F.silu(x @ w_g_cut) * (x @ w_v_cut)) @ w_o_cut
    return {"output": out, "hidden_width": h, "parameter_count": 3 * d * h}
