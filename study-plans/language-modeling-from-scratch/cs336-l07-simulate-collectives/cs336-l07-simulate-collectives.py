import torch

def simulate_collectives(rank_tensors, collective):
    r = len(rank_tensors)
    if collective == "all_gather":
        return [torch.cat(rank_tensors) for _ in range(r)]
    elif collective == "all_reduce":
        reduced_tensor = torch.stack(rank_tensors).sum(dim=0)
        return [reduced_tensor for _ in range(r)]
    elif collective == "reduce_scatter":
        reduced_tensor = torch.stack(rank_tensors).sum(dim=0)
        chunks = torch.chunk(reduced_tensor, r, dim=0)
        return [chunks[i] for i in range(r)]
    elif collective == "all_to_all":
        # (R, D) -> R*(R, D//R)
        chunks = torch.stack(rank_tensors).chunk(r, dim=1)
        return [c.flatten() for c in chunks]
    else:
        return []    