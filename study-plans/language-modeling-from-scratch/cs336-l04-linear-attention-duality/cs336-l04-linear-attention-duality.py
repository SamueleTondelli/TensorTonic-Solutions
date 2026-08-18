import torch

def linear_attention_duality(q, k, v):
    b, s, dk = q.shape
    dv = v.shape[-1]
    rec_state = torch.zeros((b, dk, dv), dtype=torch.float32, device=q.device)
    rec_out = torch.zeros((b, s, dv), dtype=torch.float32, device=q.device)
    for t in range(s):
        # (b, 1, dk)
        q_t = q[:, t, :].to(torch.float32).unsqueeze(1)
        # (b, 1, dk)
        k_t = k[:, t, :].to(torch.float32).unsqueeze(1)
        # (b, 1, dv)
        v_t = v[:, t, :].to(torch.float32).unsqueeze(1)
        
        rec_state += k_t.transpose(2, 1) @ v_t
        rec_out[:, t, :] = (q_t @ rec_state).squeeze(1)

    final_state = rec_state.to(q.dtype)
    rec_out = rec_out.to(q.dtype)

    # (b, s, dk, 1)
    k_exp = k.unsqueeze(-1).to(torch.float32)
    # (b, s, 1, dv)
    v_exp = v.unsqueeze(2).to(torch.float32)
    # (b, s, dk, dv)
    par_states = (k_exp @ v_exp).cumsum(1)
    # (b, s, 1, dk)
    q_exp = q.unsqueeze(2).to(torch.float32)
    # (b, s, dv)
    par_out = (q_exp @ par_states).squeeze(2).to(q.dtype)
    
    return {"parallel_output": par_out, "recurrent_output": rec_out, "final_state": final_state}
