import numpy as np
from typing import List, Any
from utils import io
from utils.helper import get_twin, locate_twin, is_schedule_violated
from globals import SLOTS_PER_DAY, TOTAL_DURATION

class ConstraintChecker:
    def __init__(
            self, 
            chromosome: np.ndarray, 
            row_indices: List[int] = [], 
            col_indices: List[int] = [],
            verbose: bool = False
        ):
        self.chromosome = chromosome

        T, R = chromosome.shape
        self.row_indices = list(range(T)) if not row_indices else row_indices
        self.col_indices = list(range(R)) if not col_indices else col_indices

        self.verbose = verbose
        self.faulty: List[Any] = []

    def check_frequencies(self) -> bool:
        return np.count_nonzero(self.chromosome) == TOTAL_DURATION

    def subject_session_per_day_find(self) -> List[int]:
        self.faulty = []
        arr = self.chromosome

        for i in self.row_indices:
            for j in self.col_indices:
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
                i, j = self.faulty[0]
                val = self.chromosome[i, j]
                print(f"Multiple sessions of subject in a day of val: {val}")
            return False
        
        return True
    
    def subject_session_per_day_fix(self):
        self.subject_session_per_day_find()
        arr = self.chromosome

        for row, col in self.faulty:
            val = arr[row, col]
            twin_location = locate_twin(arr, val)
            if twin_location is None:
                raise Exception("No twin found")

            row_twin, col_twin = twin_location
            
            placed = False
            for i in self.row_indices:
                if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                    continue

                for j in self.col_indices:
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

    def time_constraint_find(self):
        self.faulty = []
        arr = self.chromosome

        for i in self.row_indices:
            for j in self.col_indices:
                val = arr[i, j]
                if val == 0:
                    continue

                _arr = arr.copy() // 100
                if is_schedule_violated(_arr[i].flatten(), val // 100):
                    self.faulty.append({
                        "val": int(val),
                        "location": (i, j)
                    })
                    arr[i, j] = 0

        self.chromosome = arr

    def time_constraint_check(self) -> bool:
        self.time_constraint_find()
        if self.faulty:
            if self.verbose:
                i, j = self.faulty[0]["location"]
                val = self.faulty[0]["val"]
                print(f"Time constraint violation of val: {val} in row: {i}")
            return False
        
        return True
    
    def time_constraint_fix(self):
        self.time_constraint_find()
        arr = self.chromosome

        for el in self.faulty:
            val = el["val"]
            row, col = el["location"]
            twin_location = locate_twin(arr, val)

            placed = False
            for i in self.row_indices:
                if twin_location is not None:
                    row_twin, col_twin = twin_location
                    if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                        continue
                
                _arr = arr.copy() // 100
                if is_schedule_violated(_arr[i].flatten(), val // 100):
                    continue

                for j in self.col_indices:
                    if arr[i, j] == 0:
                        arr[i, j] = val
                        placed = True
                        break

                if placed:
                    break
            
            if not placed:
                # Error may occur here
                # Let's hope this genome won't be chosen for the next gen
                io.export_to_txt(arr, "debug", f"error_child.txt")
                        
        self.chromosome = arr

    def validate(self) -> bool:
        if not self.check_frequencies():
            if self.verbose:
                print("Constraint failed: Frequency does not match")
            return False
        if not self.subject_session_per_day_check():
            return False
        if not self.time_constraint_check():
            if self.verbose:
                print("Constraint failed: Time constraint")
            return False
        return True
