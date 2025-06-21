import numpy as np
from globals import Configuration
from utils.helper import get_adjacent_classes, haversine
from ga import generator
from ga.constraint_checker import ConstraintChecker
from ga.mutation_operator import MutationOperator
from ga.parallel_class import ParallelClass
from dataframes.curriculum import Curriculum
from typing import List, Optional, Union

class Genome:

    def __init__(self, chromosome: np.ndarray, config: Configuration):
        self.chromosome = chromosome
        self.config = config

        self.cached_check_constraint: Optional[bool] = None
        self.cached_config: Optional[List[np.ndarray]] = None
        self.cached_used_rooms: Optional[int] = None
        self.cached_average_distance: Optional[float] = None
        self.cached_average_size: Optional[float] = None
        
        self.rank: Optional[int] = None
        self.dominated_set: List[Genome] = []
        self.domination_count: int = 0
        self.crowding_distance: float = 0.0

    @classmethod
    def from_generator(
        cls,
        curriculum: Curriculum,
        time_slot_indices: List,
        room_indices: List,
        config: Configuration,
    ):
        guess = generator.generate_valid_guess(curriculum, time_slot_indices, room_indices, config)
        return cls(guess, config)
    
    def reset_state(self):
        self.rank = None
        self.dominated_set = []
        self.domination_count = 0
        self.crowding_distance = 0.0

    def clear_cache(self):
        self.cached_check_constraint = None
        self.cached_config = None
        self.cached_used_rooms = None
        self.cached_average_distance = None
        self.cached_average_size = None

    def get_objectives(self) -> List[Union[int, float]]:
        return np.array([self.calculate_average_distance(), self.calculate_average_size()])
    
    # def count_used_rooms(self) -> int:
    #     if self.cached_used_rooms is not None:
    #         return self.cached_used_rooms
        
    #     if self.check_constraint(verbose=True):
    #         result = np.count_nonzero(np.any(self.chromosome != 0, axis=0))
    #         self.cached_used_rooms = result
    #         return result
    #     else:
    #         return 1000
        
    def calculate_average_distance(self) -> float:
        if self.cached_average_distance is not None:
            return self.cached_average_distance
        
        if self.cached_check_constraint is None:
            self.cached_check_constraint = self.check_constraint(verbose=True)
        
        if not self.cached_check_constraint:
            return 1000.0
                
        config = self.get_config()
        results = []
        for c in config:
            adjacents = get_adjacent_classes(c)
            distances = []
            for adj in adjacents:
                room1, room2 = adj
                point1 = self.config.coordinates[room1 + 1]
                point2 = self.config.coordinates[room2 + 1]
                distance = haversine(point1, point2)
                distances.append(distance)
            
            avg = sum(distances) / len(distances) if distances else 0.0
            results.append(avg)

        result_value = sum(results) / len(results) if results else 0.0
        self.cached_average_distance = result_value
        return result_value
    
    def calculate_average_size(self) -> float:
        if self.cached_average_size is not None:
            return self.cached_average_size
        
        if self.cached_check_constraint is None:
            self.cached_check_constraint = self.check_constraint(verbose=True)
        
        if not self.cached_check_constraint:
            return 0
        
        config = self.get_config()
        for c in config:
            rooms = np.nonzero(c)[1].tolist()
            if not rooms:
                raise ValueError("No rooms found in the configuration")

            sizes = [self.config.sizes[room + 1] for room in rooms]
            return sum(sizes) / len(sizes)

    def check_constraint(self, verbose):
        return ConstraintChecker(self.chromosome, self.config, verbose=verbose).validate()
    
    def mutate(self):
        self.chromosome = MutationOperator(self.chromosome).mutate()
        self.reset_state()
        self.clear_cache()
    
    def get_config(self) -> List[np.ndarray]:
        if self.cached_config is None:
            config = ParallelClass(self.chromosome, self.config).get_all_schedule_matrices()
            self.cached_config = config
            return config
        else:
            return self.cached_config