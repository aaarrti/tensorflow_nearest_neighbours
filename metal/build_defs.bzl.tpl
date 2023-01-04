def metal_is_configured():
    return %{metal_is_configured}

def if_metal_is_configured(x):
    if metal_is_configured():
        return x
    else:
        return []