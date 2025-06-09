import numpy as np
from globals import COORDINATES
from utils.helper import get_adjacent_classes, haversine
from ga import generator
from ga.constraint_checker import ConstraintChecker
from ga.mutation_operator import MutationOperator
from ga.constraint_checker import ConstraintChecker
from ga.parallel_class import ParallelClass
from dataframes.curriculum import Curriculum
from typing import List, Optional, Union

class Genome:

    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome

        self.cached_config: List[np.ndarray] = []
        self.cached_used_rooms: Optional[int] = None
        self.cached_average_distance: Optional[float] = None
        
        self.rank: Optional[int] = None
        self.dominated_set: List[Genome] = []
        self.domination_count: int = 0
        self.crowding_distance: float = 0.0

    @classmethod
    def from_generator(cls, curriculum: Curriculum, time_slot_indices: List, room_indices: List):
        guess = generator.generate_valid_guess(curriculum, time_slot_indices, room_indices)
        return cls(guess)
    
    def reset_state(self):
        self.rank = None
        self.dominated_set = []
        self.domination_count = 0
        self.crowding_distance = 0.0

    def clear_cache(self):
        self.cached_config = []
        self.cached_used_rooms = None
        self.cached_average_distance = None
    
    def count_used_rooms(self) -> int:
        if self.cached_used_rooms is not None:
            return self.cached_used_rooms
        
        if self.check_constraint(verbose=True):
            result = np.count_nonzero(np.any(self.chromosome != 0, axis=0))
            self.cached_used_rooms = result
            return result
        else:
            return 1000
        
    def get_objectives(self) -> List[Union[int, float]]:
        return np.array([self.count_used_rooms(), self.calculate_average_distance()])
        
    def calculate_average_distance(self) -> float:
        if self.cached_average_distance is not None:
            return self.cached_average_distance
        
        config = self.get_config()
        results = []
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
            results.append(avg)

        result_value = sum(results) / len(results) if results else 0.0
        self.cached_average_distance = result_value
        return result_value

    def check_constraint(self, verbose):
        return ConstraintChecker(self.chromosome, verbose=verbose).validate()
    
    def mutate(self):
        return MutationOperator(self.chromosome).mutate()
    
    def get_config(self) -> List[np.ndarray]:
        if self.cached_config == []:
            pc = ParallelClass(self.chromosome)
            return pc.get_all_schedule_matrices()
        else:
            return self.cached_config