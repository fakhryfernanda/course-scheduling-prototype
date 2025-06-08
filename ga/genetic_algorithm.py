import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import io
from globals import *
from typing import List
from ga.genome import Genome
from ga.crossover_operator import CrossoverOperator
from ga.parent_selection import ParentSelection
from ga.parallel_class import ParallelClass
from dataclasses import dataclass
from dataframes.curriculum import Curriculum

@dataclass
class ProblemContext:
    curriculum: Curriculum
    time_slot_indices: List[int]
    room_indices: List[int]

@dataclass
class FitnessStats:
    best: float
    worst: float
    average: float

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
        self.generation: int = 0

        self.room_count_fitness: dict[str, FitnessStats] = {}
        self.average_distance_fitness: dict[str, FitnessStats] = {}
        self.best_genome: Genome

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
        used_rooms = [
            genome.count_used_rooms()
            for genome in self.population
        ]

        self.room_count_fitness[self.generation] = FitnessStats(
            best=min(used_rooms), 
            worst=max(used_rooms), 
            average=sum(used_rooms) / self.population_size
        )

        return used_rooms

    def plot_evaluation(self):
        eval = self.room_count_fitness

        x = list(eval.keys())
        best = [eval[i].best for i in x]
        worst = [eval[i].worst for i in x]
        average = [eval[i].average for i in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, best, label='Best Fitness')
        plt.plot(x, worst, label='Worst Fitness')
        plt.plot(x, average, label='Average Fitness')

        info_text = (
            f"Population size: {self.population_size}\n"
            f"Crossover rate: {CROSSOVER_RATE}\n"
            f"Mutation rate: {MUTATION_RATE}\n"
            f"Mutation points: {MUTATION_POINTS}"
        )
        plt.text(0.01, 0.02, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evaluation Over Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to file with timestamp
        os.makedirs("evaluation", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"evaluation/{timestamp}.png")

        plt.show()
    
    def validate(self):
        return [
            genome.check_constraint()
            for genome in self.population
        ]

    def print_parallel_class_config(self, genome: Genome) -> None:
        config = genome.get_config()
        
        for idx, mat in enumerate(config, 1):
            combo = tuple(((idx - 1) % n) + 1 for n in PARALLEL_COUNTS)
            print(f"\nConfiguration {idx} {combo}:\n{mat}")
        
    def select(self):
        return ParentSelection(method=SELECTION_METHOD).run(self.population)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        return CrossoverOperator().run(parent1, parent2)

    def evolve(self):
        next_population = []

        # Selection
        parents = self.select()

        # Crossover
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1]

            identical = np.array_equal(p1.chromosome, p2.chromosome)
            if random.random() < CROSSOVER_RATE and not identical:
                child1 = self.crossover(p1.chromosome, p2.chromosome)
                child2 = self.crossover(p2.chromosome, p1.chromosome)
                next_population.append(Genome(child1))
                next_population.append(Genome(child2))
            else:
                next_population.extend([Genome(p1.chromosome), Genome(p2.chromosome)])

        # Mutation
        for genome in next_population:
            if random.random() < MUTATION_RATE:
                genome.mutate()

        self.population = next_population
        self.eval()
        self.best_genome = min(self.population, key=lambda g: g.count_used_rooms())
        self.generation += 1

    def run(self):
        for _ in range(MAX_GENERATION):
            self.evolve()
            if self.generation > 0.95 * MAX_GENERATION:
                self.export_population()