import numpy as np
from globals import COORDINATES
from utils.helper import get_adjacent_classes, haversine
from ga import generator
from ga.constraint_checker import ConstraintChecker
from ga.mutation_operator import MutationOperator
from ga.constraint_checker import ConstraintChecker
from ga.parallel_class import ParallelClass
from dataframes.curriculum import Curriculum
from typing import List

class Genome:
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome

    @classmethod
    def from_generator(cls, curriculum: Curriculum, time_slot_indices: List, room_indices: List):
        guess = generator.generate_valid_guess(curriculum, time_slot_indices, room_indices)
        return cls(guess)
    
    def count_used_rooms(self) -> int:
        if self.check_constraint(verbose=True):
            return np.count_nonzero(np.any(self.chromosome != 0, axis=0))
        else:
            return 1000
        
    def calculate_average_distance(self) -> float:
        config = self.get_config()
        result = []
        for c in config:
            adjacents = get_adjacent_classes(c)
            distances = []
            for adj in adjacents:
                room1, room2 = adj
                point1 = COORDINATES[room1+1]
                point2 = COORDINATES[room2+1]
                distance = haversine(point1, point2)
                distances.append(distance)
            
            avg = sum(distances) / len(distances) if distances else 0.0
            result.append(avg)

        return sum(result) / len(result) if result else 0.0

    def check_constraint(self, verbose):
        return ConstraintChecker(self.chromosome, verbose=verbose).validate()
    
    def mutate(self):
        return MutationOperator(self.chromosome).mutate()
    
    def get_config(self) -> List[np.ndarray]:
        pc = ParallelClass(self.chromosome)
        return pc.get_all_schedule_matrices()