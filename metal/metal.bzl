def is_metal_enabled():
    print("is_metal_enabled->")
    return True
    #return %{metal_enabled}

def if_metal_is_enabled(x):
    if is_metal_enabled():
        return x
    return []
