import random
import numpy as np
from globals import *
from ga.constraint_checker import ConstraintChecker
from utils.helper import get_twin, locate_twin, is_schedule_violated
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
    
    def column_based_crossover_old(self, p1: np.ndarray, p2: np.ndarray):
        rows, cols = p1.shape
        midpoint = cols // 2
        child = p1.copy()
        child[:, midpoint:] = p2[:, midpoint:]

        # Step 1: Count values
        parent_counter = Counter(p1.flatten())
        child_counter = Counter(child.flatten())

        # Step 2: Find duplicate positions in right half
        duplicate_positions = []
        for i in range(rows):
            for j in range(midpoint, cols):
                val = child[i, j]
                if val != 0 and child_counter[val] > 1:
                    duplicate_positions.append((i, j))

        # Step 3: Find missing values
        missing = list(set(parent_counter.keys()) - set(child_counter.keys()))
        missing = [val for val in missing if val != 0]
        random.shuffle(missing)
        random.shuffle(duplicate_positions)

        # Step 4: Replace duplicates with missing values
        used = min(len(duplicate_positions), len(missing))
        for k in range(used):
            i, j = duplicate_positions[k]
            child[i, j] = missing[k]

        # Step 5: Remove leftover duplicates
        for i, j in duplicate_positions[used:]:
            child[i, j] = 0

        # Step 6: Fill leftover missing values randomly
        empty_positions = [(i, j) for j in range(midpoint, cols)
                                for i in range(rows) if child[i, j] == 0]
        random.shuffle(empty_positions)
        for val in missing[used:]:
            if not empty_positions:
                raise Exception("No space left to insert missing values")
            i, j = empty_positions.pop()
            child[i, j] = val

        # Step 7: Frequency check
        child_counter = Counter(child.flatten())
        if child_counter != parent_counter:
            raise Exception("Crossover failed")
        
        # Step 8: Faulty check
        faulty = []
        for i in range(rows):
            for j in range(midpoint, cols):
                val = child[i, j]
                if val == 0:
                    continue

                twin_val = get_twin(child, val)
                if twin_val is None:
                    continue

                twin_location = locate_twin(child, val)
                if twin_location is None:
                    continue

                row_twin, col_twin = twin_location
                if (row_twin, col_twin) in faulty:
                    continue

                if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                    faulty.append((i, j))

        # Step 9: Fix faulty
        for row, col in faulty:
            val = child[row, col]
            twin_location = locate_twin(child, val)
            if twin_location is None:
                raise Exception("No twin found")

            row_twin, col_twin = twin_location
            
            placed = False
            for i in range(rows):
                if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                    continue

                for j in range(midpoint, cols):
                    if child[i, j] == 0:
                        child[i, j] = val
                        child[row, col] = 0
                        placed = True
                        break

                if placed:
                    break
            
            if not placed:
                raise Exception("No space left to fix faulty")

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

        # Frequency check
        child_counter = Counter(child.flatten())
        if child_counter != parent_counter:
            raise Exception("Crossover failed")
        
        # Faulty check (subject session)    
        checker = ConstraintChecker(child)
        checker.subject_session_per_day_fix(list(range(rows)), list(range(midpoint, cols)))
        child = checker.chromosome

        # Faulty check time constraint violation
        # faulty = []
        # for i in range(rows):
        #     for j in range(midpoint, cols):
        #         val = child[i, j]
        #         if val == 0:
        #             continue

        #         if is_schedule_violated(child[i].flatten(), val):
        #             faulty.append((i, j))
        #             continue

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