import random
import numpy as np
from globals import *
from collections import Counter

class CrossoverOperator:
    def __init__(self):
        pass

    def run(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if CROSSOVER_METHOD == "column_based":
            child = self.column_based_crossover(parent1, parent2)
        elif CROSSOVER_METHOD == "row_based":
            child = self.row_based_crossover(parent1, parent2)
        else:
            raise Exception("Invalid crossover method")

        return child
    
    def column_based_crossover(self, p1: np.ndarray, p2: np.ndarray):
        rows, cols = p1.shape
        midpoint = cols // 2

        child = p1.copy()
        child[:, midpoint:] = p2[:, midpoint:]

        # Find duplicates
        child_counter = Counter(child.flatten())

        for val, freq in child_counter.items():
            if val == 0 or freq <= 1:
                continue

            removed = False
            for j in range(midpoint, cols):
                for i in range(rows):
                    if child[i, j] == val:
                        child[i, j] = 0
                        removed = True
                        break
                if removed:
                    break

        # Find missing values
        parent_counter = Counter(p1.flatten())
        child_counter = Counter(child.flatten())

        missing = list(set(parent_counter.keys()) - set(child_counter.keys()))
        missing = [val for val in missing if val != 0]
        random.shuffle(missing)

        # Collect all empty positions in bottom half
        empty_positions = [(i, j) for j in range(midpoint, cols)
                                for i in range(rows) if child[i, j] == 0]
        random.shuffle(empty_positions)

        # Insert missing values
        for val in missing:
            if not empty_positions:
                raise Exception("No space left to insert missing values")
            i, j = empty_positions.pop()
            child[i, j] = val

        # Final check
        child_counter = Counter(child.flatten())
        if child_counter != parent_counter:
            raise Exception("Crossover failed")

        return child
    
    def row_based_crossover(self, p1: np.ndarray, p2: np.ndarray):
        rows, cols = p1.shape
        midpoint = rows // 2

        child = p1.copy()
        child[midpoint:, :] = p2[midpoint:, :]

        # Find duplicates
        child_counter = Counter(child.flatten())

        for val, freq in child_counter.items():
            if val == 0 or freq <= 1:
                continue

            removed = False
            for i in range(midpoint, rows):
                for j in range(cols):
                    if child[i, j] == val:
                        child[i, j] = 0
                        removed = True
                        break
                if removed:
                    break

        # Find missing values
        parent_counter = Counter(p1.flatten())
        child_counter = Counter(child.flatten())

        missing = list(set(parent_counter.keys()) - set(child_counter.keys()))
        missing = [val for val in missing if val != 0]
        random.shuffle(missing)

        # Collect all empty positions in bottom half
        empty_positions = [(i, j) for i in range(midpoint, rows)
                                for j in range(cols) if child[i, j] == 0]
        random.shuffle(empty_positions)

        # Insert missing values
        for val in missing:
            if not empty_positions:
                raise Exception("No space left to insert missing values")
            i, j = empty_positions.pop()
            child[i, j] = val

        # Final check
        child_counter = Counter(child.flatten())
        if child_counter != parent_counter:
            raise Exception("Crossover failed")
        
        return child