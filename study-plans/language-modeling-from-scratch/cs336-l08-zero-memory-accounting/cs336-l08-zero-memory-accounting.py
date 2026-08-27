import math

def zero_memory_accounting(parameter_bytes, gradient_bytes, optimizer_bytes, world_size):
    """
    Returns: dictionary containing DDP and ZeRO byte totals
    """
    return {
        "ddp": parameter_bytes + gradient_bytes + optimizer_bytes,
        "zero1": parameter_bytes + gradient_bytes + math.ceil(optimizer_bytes/world_size),
        "zero2": parameter_bytes + math.ceil((gradient_bytes + optimizer_bytes)/world_size),
        "zero3": math.ceil((parameter_bytes + gradient_bytes + optimizer_bytes)/world_size)
    }
