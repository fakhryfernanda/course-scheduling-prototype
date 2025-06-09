import os
import ast
import numpy as np

def export_to_txt(arr, folder="solutions", filename="solution.txt"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Full file path
    filepath = os.path.join(folder, filename)

    # Convert array to string without clipping or wrapping
    with np.printoptions(threshold=np.inf, linewidth=10000):
        arr_str = np.array2string(arr, separator=', ')

    # Count non-zero values
    non_zeros = np.count_nonzero(arr)

    # Write to file
    with open(filepath, 'w') as f:
        f.write(arr_str + "\n")
        f.write(f"\nNon-zero values: {non_zeros}\n")

def import_from_txt(folder="solutions", filename="solution.txt"):
    filepath = os.path.join(folder, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract lines until a blank or metadata line
    array_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" or "Non-zero values:" in stripped or "Element Frequencies:" in stripped:
            break
        array_lines.append(stripped)

    if not array_lines:
        raise ValueError("No array content found in file.")

    array_str = " ".join(array_lines)

    try:
        parsed = ast.literal_eval(array_str)
        arr = np.array(parsed, dtype=np.int16)
    except Exception as e:
        raise ValueError(f"Failed to parse array from file: {e}")

    return arr

def import_all_txt_arrays(folder):
    arrays = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            try:
                arr = import_from_txt(folder, filename)
                arrays.append(arr)
            except Exception as e:
                print(f"Failed to import '{filename}': {e}")

    return arrays
