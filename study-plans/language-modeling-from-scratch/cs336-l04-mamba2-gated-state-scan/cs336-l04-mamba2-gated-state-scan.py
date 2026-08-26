import torch

def mamba2_gated_scan(q, k, v, gamma):
    b, s, d_k = q.shape
    d_v = v.shape[-1]

    state = torch.zeros((b, d_k, d_v), dtype=torch.float32, device=q.device)
    outputs = torch.zeros((b, s, d_v), dtype=torch.float32, device=q.device)
    for t in range(s):
        # (B, 1, d_*)
        q_t = q[:, t, :].unsqueeze(1).to(torch.float32)
        k_t = k[:, t, :].unsqueeze(1).to(torch.float32)
        v_t = v[:, t, :].unsqueeze(1).to(torch.float32)

        # (B, 1, 1)
        gamma_t = gamma[:, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)

        # (B, d_k, d_v)
        state = gamma_t * state + k_t.transpose(2, 1) @ v_t
        # (B, 1, d_v)
        out_t = q_t @ state
        outputs[:, t, :] = out_t.squeeze(1)
    return {"outputs": outputs.to(q.dtype), "final_state": state.to(q.dtype)}    
