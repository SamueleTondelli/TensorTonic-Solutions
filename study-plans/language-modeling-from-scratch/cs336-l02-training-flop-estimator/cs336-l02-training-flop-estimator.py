def flop_estimator(matmuls, attention_flops=0):
    """
    Returns: dictionary containing exact forward, backward, and total FLOP counts
    """
    fwd_flops = attention_flops
    for m in matmuls:
        fwd_flops += 2 * m[0] * m[1] * m[2]
    bwd_flops = 2 * fwd_flops
    return {"forward_flops": fwd_flops, "backward_flops": bwd_flops, "total_flops": fwd_flops + bwd_flops}
