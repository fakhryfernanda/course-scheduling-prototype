import numpy as np
from globals import SLOTS_PER_DAY, TOTAL_DURATION

class ConstraintChecker:
    def __init__(self, chromosome: np.ndarray, verbose: bool = False):
        self.chromosome = chromosome
        self.verbose = verbose

    def check_frequencies(self) -> bool:
        return np.count_nonzero(self.chromosome) == TOTAL_DURATION

    def check_subject_session_per_day(self) -> bool:
        T, R = self.chromosome.shape
        days_count = T // SLOTS_PER_DAY
        for day in range(days_count):
            start = day * SLOTS_PER_DAY
            end = start + SLOTS_PER_DAY
            for t in range(start, end):
                seen_keys = set()
                for r in range(R):
                    val = self.chromosome[t, r]
                    if val == 0:
                        continue
                    subject_id = val // 100
                    class_number = val // 10 % 10
                    key = (subject_id, class_number)
                    if key in seen_keys:
                        if self.verbose:
                            print(f"Multiple sessions of subject {subject_id} class {class_number} on day {day}")
                        return False
                    seen_keys.add(key)
        return True

    def validate(self) -> bool:
        if not self.check_frequencies():
            if self.verbose:
                print("Constraint failed: Frequency does not match")
            return False
        if not self.check_subject_session_per_day():
            if self.verbose:
                print("Constraint failed: Subject session per day check")
            return False
        return True
