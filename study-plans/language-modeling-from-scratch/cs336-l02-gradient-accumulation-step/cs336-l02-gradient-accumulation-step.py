import torch

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    n = len(microbatch_inputs)
    param = param.detach().clone().requires_grad_()
    n_ins = sum([microbatch_targets[i].shape[0] for i in range(n)])
    for i in range(n):
        inp = microbatch_inputs[i].detach().clone().requires_grad_()
        tgt = microbatch_targets[i].detach().clone().requires_grad_()
        L = (inp @ param - tgt).square().sum() / n_ins
        L.backward()
    return {"new_param": param - lr * param.grad, "full_grad": param.grad}
