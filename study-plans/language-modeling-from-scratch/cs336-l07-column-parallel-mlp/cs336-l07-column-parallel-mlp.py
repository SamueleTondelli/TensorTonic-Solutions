import torch

def column_parallel_mlp(x, weight_shards, bias_shards=None):
    """
    Returns: dictionary containing local and full activations
    """
    local_activations = []
    for i in range(len(weight_shards)):
        local_activations.append(x @ weight_shards[i])
        if bias_shards is not None:
            local_activations[i] += bias_shards[i]
    return {"local_activations": local_activations, "full_activation": torch.cat(local_activations, dim=-1)}
