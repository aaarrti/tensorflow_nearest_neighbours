def is_cuda_enabled():
    return True
    #return %{cuda_enabled}

def if_cuda_is_enabled(x):
    if is_cuda_enabled():
        return x
    return []
