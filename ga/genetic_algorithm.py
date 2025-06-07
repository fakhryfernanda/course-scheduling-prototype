import numpy as np
from utils import io
from globals import *
from typing import List
from ga.genome import Genome
from ga.crossover_operator import CrossoverOperator
from dataclasses import dataclass
from dataframes.curriculum import Curriculum

@dataclass
class ProblemContext:
    curriculum: Curriculum
    time_slot_indices: List[int]
    room_indices: List[int]

class GeneticAlgorithm:
    def __init__(
            self, 
            context: ProblemContext, 
            population_size: int, 
        ):

        self.context = context

        assert population_size % 2 == 0, "Population size must be even"
        self.population_size = population_size
        self.population: List[Genome] = []
        self.generation = 0

        self.initialize_population()

    def initialize_population(self):
        self.population = [
            Genome.from_generator(
                self.context.curriculum,
                self.context.time_slot_indices,
                self.context.room_indices
            )
            for _ in range(self.population_size)
        ]
        self.export_population()

    def export_population(self):
        for i, genome in enumerate(self.population):
            io.export_to_txt(genome.chromosome, f"population/gen_{self.generation}", f"p_{i+1}.txt")

    def eval(self):
        pass
    
    def validate(self):
        return [
            genome.check_constraint()
            for genome in self.population
        ]
        
    def select(self):
        pass

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        return CrossoverOperator().run(parent1, parent2)

    def evolve(self):
        pass

    def run(self):
        pass
