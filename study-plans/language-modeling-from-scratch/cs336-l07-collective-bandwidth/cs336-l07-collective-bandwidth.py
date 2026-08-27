def collective_bandwidth(payload_bytes, world_size, duration_seconds, collective):
    c = 2 if collective == "all_reduce" else 1
    a = c * payload_bytes * (world_size - 1) / (world_size)
    return {
        "algorithm_bytes": a,
        "bandwidth_bytes_per_second": a / duration_seconds
    }
