import torch

def rmsnorm(x, g, epsilon):
    x_f32 = x.to(torch.float32)
    d = x_f32.shape[-1]
    sq_mean = x_f32.square().sum(-1, keepdim=True) / d
    x_norm = x_f32 / torch.sqrt(sq_mean + epsilon)
    return x_norm.to(x.dtype) * g
    