import numpy as np
from globals import SLOTS_PER_DAY

def locate_value(arr, val):
    result = np.argwhere(arr == val)
    return result[0] if len(result) > 0 else None

def get_twin(arr, val):
    twin_val = val + 1 if val % 10 == 1 else val - 1
    return twin_val if twin_val in arr else None

def locate_twin(arr, val):
    twin_val = get_twin(arr, val)
    return locate_value(arr, twin_val) if twin_val else None

def get_adjacent_classes(arr):
    T, R = arr.shape
    adjacent_classes = []

    for i in range(T):
        if (i + 1) % SLOTS_PER_DAY == 0:
            continue

        for j in range(R):
            curr = arr[i, j]
            next_ = arr[i + 1, j]

            if curr == 0 or next_ == 0:
                continue

            adjacent_classes.append((int(curr), int(next_)))

    return adjacent_classes