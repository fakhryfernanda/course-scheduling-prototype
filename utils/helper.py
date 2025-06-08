import numpy as np

def locate_value(arr, val):
    result = np.argwhere(arr == val)
    return result[0] if len(result) > 0 else None

def get_twin(arr, val):
    twin_val = val + 1 if val % 10 == 1 else val - 1
    return twin_val if twin_val in arr else None

def locate_twin(arr, val):
    twin_val = get_twin(arr, val)
    return locate_value(arr, twin_val) if twin_val else None