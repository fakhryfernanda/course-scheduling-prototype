import math
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

def is_schedule_violated(arr, val):    
    return np.any(arr) and np.any((arr != 0) & (arr != val))

def get_adjacent_classes(arr):
    # Only for parallel class config
    nonzero_counts = np.count_nonzero(arr, axis=1)

    if np.any(nonzero_counts > 1):
        bad_rows = np.where(nonzero_counts > 1)[0]
        raise ValueError(f"Row(s) with more than one non-zero value: {bad_rows}")

    nonzero_indices = np.where(arr != 0, np.arange(arr.shape[1]), -1)
    first_nonzero_per_row = np.max(nonzero_indices, axis=1)

    result = []
    for i in range(len(first_nonzero_per_row) - 1):
        if (i+1) % SLOTS_PER_DAY == 0:
            continue

        a = first_nonzero_per_row[i]
        b = first_nonzero_per_row[i + 1]
        if a != -1 and b != -1:
            result.append((int(a), int(b)))

    return result

def haversine(point1: tuple[float, float], point2: tuple[int, int]):
    R = 6371000  # Earth radius in meters

    lat1, long1 = point1
    lat2, long2 = point2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(long2 - long1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
        

