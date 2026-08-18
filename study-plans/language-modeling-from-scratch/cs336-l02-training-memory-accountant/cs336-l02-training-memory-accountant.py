def get_n(shape):
    n = 0
    for s in shape:
        n_t = 1
        for d in s:
            n_t *= d
        n += n_t
    return n

def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element,
                       activation_shapes, activation_bytes_per_element,
                       optimizer, optimizer_bytes_per_element):
    n_param = get_n(param_shapes)
    param_bytes = n_param * param_bytes_per_element
    grad_bytes = n_param * grad_bytes_per_element

    n_activation = get_n(activation_shapes)
    activation_bytes = n_activation * activation_bytes_per_element

    optimizer_bytes = 0
    if optimizer == "adam":
        optimizer_bytes = n_param * 2 * optimizer_bytes_per_element
    elif optimizer == "adagrad":
        optimizer_bytes = n_param * optimizer_bytes_per_element

    return {"parameters": param_bytes, "gradients": grad_bytes, "activations": activation_bytes, "optimizer_state": optimizer_bytes, "total": param_bytes + grad_bytes + activation_bytes + optimizer_bytes}
    
