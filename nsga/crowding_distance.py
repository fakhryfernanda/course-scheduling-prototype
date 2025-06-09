from typing import List
from ga.genome import Genome

class CrowdingDistance:
    def __init__(self, front: List[Genome]):
        self.front = front

    def assign(self):
        n = len(self.front)
        if n == 0:
            return
        elif n == 1:
            self.front[0].crowding_distance = float('inf')
            return
        elif n == 2:
            self.front[0].crowding_distance = float('inf')
            self.front[1].crowding_distance = float('inf')
            return

        # Reset
        for genome in self.front:
            genome.crowding_distance = 0.0

        objectives = [
            Genome.calculate_average_distance,
            Genome.calculate_average_size
        ]

        for obj_func in objectives:
            self.front.sort(key=obj_func)

            self.front[0].crowding_distance = float('inf')
            self.front[-1].crowding_distance = float('inf')

            f_min = obj_func(self.front[0])
            f_max = obj_func(self.front[-1])
            norm = f_max - f_min if f_max != f_min else 1e-9

            for i in range(1, n - 1):
                prev = obj_func(self.front[i - 1])
                next_ = obj_func(self.front[i + 1])
                self.front[i].crowding_distance += (next_ - prev) / norm
