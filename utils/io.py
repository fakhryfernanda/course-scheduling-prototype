import os
import ast
import numpy as np
from collections import Counter

def export_to_txt(arr, folder="solutions", filename="solution.txt"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Full file path
    filepath = os.path.join(folder, filename)

    # Convert array to string without clipping or wrapping
    with np.printoptions(threshold=np.inf, linewidth=10000):
        arr_str = np.array2string(arr, separator=', ')

    # Write to file
    with open(filepath, 'w') as f:
        f.write(arr_str + "\n\n")

def import_from_txt(folder="output", filename="output.txt"):
    filepath = os.path.join(folder, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract only the array part
    array_lines = []
    array_started = False

    for line in lines:
        if line.strip() == "Array:":
            array_started = True
            continue
        if array_started:
            if line.strip() == "" or line.strip().startswith("Element Frequencies:"):
                break
            array_lines.append(line.strip())

    array_str = "".join(array_lines)
    
    try:
        parsed = ast.literal_eval(array_str)
        arr = np.array(parsed, dtype=np.int16)
    except Exception as e:
        raise ValueError(f"Failed to parse array from file: {e}")

    return arr