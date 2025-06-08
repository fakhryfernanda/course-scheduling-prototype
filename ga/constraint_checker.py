import numpy as np
from typing import List
from utils.helper import get_twin, locate_twin
from globals import SLOTS_PER_DAY, TOTAL_DURATION

class ConstraintChecker:
    def __init__(self, chromosome: np.ndarray, verbose: bool = False):
        self.chromosome = chromosome
        self.verbose = verbose
        self.faulty: List[int] = []

    def check_frequencies(self) -> bool:
        return np.count_nonzero(self.chromosome) == TOTAL_DURATION

    def subject_session_per_day_find(self, row_indices: List[int] = [], col_indices: List[int] = []) -> List[int]:
        arr = self.chromosome
        T, R = arr.shape
        if not row_indices:
            row_indices = range(T)
        if not col_indices:
            col_indices = range(R)

        for i in row_indices:
            for j in col_indices:
                val = arr[i, j]
                if val == 0:
                    continue

                twin_val = get_twin(arr, val)
                if twin_val is None:
                    continue

                twin_location = locate_twin(arr, val)
                if twin_location is None:
                    continue

                row_twin, col_twin = twin_location
                if (row_twin, col_twin) in self.faulty:
                    continue

                if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                    self.faulty.append((i, j))
            
    def subject_session_per_day_check(self) -> bool:
        self.subject_session_per_day_find()
        if self.faulty:
            if self.verbose:
                val = self.fault[0]
                print(f"Multiple sessions of subject in a day of val: {val}")
            return False
        
        return True
    
    def subject_session_per_day_fix(self, row_indices: List[int], col_indices: List[int]):
        arr = self.chromosome
        T, R = arr.shape
        if not row_indices:
            row_indices = list(range(T))
        if not col_indices:
            col_indices = list(range(R))

        self.subject_session_per_day_find(row_indices, col_indices)

        for row, col in self.faulty:
            val = arr[row, col]
            twin_location = locate_twin(arr, val)
            if twin_location is None:
                raise Exception("No twin found")

            row_twin, col_twin = twin_location
            
            placed = False
            for i in row_indices:
                if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                    continue

                for j in col_indices:
                    if arr[i, j] == 0:
                        arr[i, j] = val
                        arr[row, col] = 0
                        placed = True
                        break

                if placed:
                    break
            
            if not placed:
                raise Exception("No space left to fix faulty")
            
        self.chromosome = arr

    def validate(self) -> bool:
        if not self.check_frequencies():
            if self.verbose:
                print("Constraint failed: Frequency does not match")
            return False
        if not self.subject_session_per_day_check():
            return False
        return True
