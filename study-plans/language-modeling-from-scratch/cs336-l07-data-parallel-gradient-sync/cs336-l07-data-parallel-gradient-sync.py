import torch

def data_parallel_gradient_sync(parameter_replicas, gradient_replicas, learning_rate):
    """
    Returns: dictionary containing mean gradients and updated replicas
    """
    r = len(parameter_replicas)
    params = [p.clone() for p in parameter_replicas[0]]
    n_params = len(params)
    gradients = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    for gr in gradient_replicas:
        for i in range(n_params):
            if gr[i] is None:
                continue
            gradients[i] += gr[i].to(torch.float32)

    for i in range(n_params):
        gradients[i] = (gradients[i] / r).to(params[i].dtype)
        params[i] -= learning_rate * gradients[i]

    return {
        "mean_gradients": gradients,
        "updated_replicas": [params for _ in range(r)]
    }
        
