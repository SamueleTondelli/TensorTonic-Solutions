import torch

def rotary_embed(q, k, positions, inv_freq):
    s = positions.shape[-1]
    sigma = positions.reshape(-1, 1, s, 1) * inv_freq
    
    sin_sigma = torch.sin(sigma)
    cos_sigma = torch.cos(sigma)

    q_rotated = torch.empty_like(q)
    k_rotated = torch.empty_like(k)

    q_rotated[..., 0::2] = q[..., 0::2] * cos_sigma - q[..., 1::2] * sin_sigma
    q_rotated[..., 1::2] = q[..., 0::2] * sin_sigma + q[..., 1::2] * cos_sigma

    k_rotated[..., 0::2] = k[..., 0::2] * cos_sigma - k[..., 1::2] * sin_sigma
    k_rotated[..., 1::2] = k[..., 0::2] * sin_sigma + k[..., 1::2] * cos_sigma
    return {"q_rotated": q_rotated, "k_rotated": k_rotated}