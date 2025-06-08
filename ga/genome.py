import numpy as np
from ga import generator
from ga.constraint_checker import ConstraintChecker
from ga.mutation_operator import MutationOperator
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
        pass
    
    def check_constraint(self, verbose):
        return ConstraintChecker(self.chromosome, verbose=verbose).validate()
    
    def mutate(self):
        return MutationOperator(self.chromosome).mutate()