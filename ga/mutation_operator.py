import numpy as np
from utils.helper import get_twin, locate_value, is_schedule_violated
from globals import SLOTS_PER_DAY, MUTATION_METHOD, MUTATION_POINTS

class MutationOperator:
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome

    def mutate(self) -> np.ndarray:
        if MUTATION_METHOD == "random_swap":
            return self.random_swap()
        else:
            raise ValueError(f"Unsupported mutation method: {self.method}")
        
    def random_swap(self):
        arr = self.chromosome
        T, R = arr.shape
        time_indices = list(range(T))
        room_indices = list(range(R))

        for _ in range(MUTATION_POINTS):
            row = np.random.choice(T)
            col = np.random.choice(R)
            val = arr[row, col]

            np.random.shuffle(time_indices)
            np.random.shuffle(room_indices)

            placed = False
            for i in time_indices:
                twin_val = get_twin(arr, val)
                if twin_val is not None:
                    twin_location = locate_value(arr, twin_val)
                    if twin_location is None:
                        raise Exception("No twin found")
                    row_twin, col_twin = twin_location
                    
                    if i // SLOTS_PER_DAY == row_twin // SLOTS_PER_DAY:
                        continue

                _arr = arr.copy() // 100
                if is_schedule_violated(_arr[i].flatten(), val // 100):
                    continue
                        
                for j in room_indices:
                    if arr[i, j] != 0:
                        continue

                    arr[i, j] = val
                    arr[row, col] = 0
                    placed = True
                    break

                if placed:
                    break

        return arr