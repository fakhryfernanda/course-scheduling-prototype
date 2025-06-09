import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import io
from globals import *
from typing import List, Optional
from ga.genome import Genome
from ga.crossover_operator import CrossoverOperator
from ga.parent_selection import ParentSelection
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
            max_generation: int,
            crossover_rate: float = CROSSOVER_RATE,
            mutation_rate: float = MUTATION_RATE,
            mutation_points: int = MUTATION_POINTS,
            seed: List[np.ndarray] = None

        ):

        self.context = context
        assert population_size % 2 == 0, "Population size must be even"
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_points = mutation_points
        self.seed = seed
        self.population: List[Genome] = []
        self.generation: int = 0

        # self.room_count_fitness: dict[str, FitnessStats] = {}
        self.average_distance_fitness: dict[str, FitnessStats] = {}
        self.average_size_fitness: dict[str, FitnessStats] = {}
        self.best_genome: Genome

        self.initialize_population()

    def initialize_population(self):
        if self.seed is not None:
            T = len(self.context.time_slot_indices)
            R = len(self.context.room_indices)
            
            assert len(self.seed) == self.population_size, "Seed size must be equal to population size"
            assert all([ch.shape == (T, R) for ch in self.seed]), f"Seed shape must be {(T,R)}"
            self.population = [Genome(ch) for ch in self.seed]
        else:
            self.population = [
                Genome.from_generator(
                    self.context.curriculum,
                    self.context.time_slot_indices,
                    self.context.room_indices
                )
                for _ in range(self.population_size)
            ]
        self.export_population()

    def export_population(self, population: Optional[List[Genome]] = None, folder: str = None):
        population = self.population if population is None else population
        folder = f"population/gen_{self.generation}" if folder is None else folder
        width = len(str(self.max_generation))

        for i, genome in enumerate(population):
            filename = f"p_{i+1:0{width}d}.txt"
            io.export_to_txt(genome.chromosome, folder, filename)

    def eval(self) -> None:
        # used_rooms = [
        #     genome.count_used_rooms()
        #     for genome in self.population
        # ]

        # self.room_count_fitness[self.generation] = FitnessStats(
        #     best=min(used_rooms), 
        #     worst=max(used_rooms), 
        #     average=sum(used_rooms) / self.population_size
        # )

        average_distances = [
            genome.calculate_average_distance()
            for genome in self.population
        ]

        self.average_distance_fitness[self.generation] = FitnessStats(
            best=min(average_distances),
            worst=max(average_distances),
            average=sum(average_distances) / self.population_size
        )

        average_sizes = [
            genome.calculate_average_size()
            for genome in self.population
        ]

        self.average_size_fitness[self.generation] = FitnessStats(
            best=max(average_sizes),
            worst=min(average_sizes),
            average=sum(average_sizes) / self.population_size
        )        

    def plot_evaluation(self, type, folder: str = None, filename:str = None):
        if type == "room_count":
            eval = self.room_count_fitness
        elif type == "average_distance":
            eval = self.average_distance_fitness
        elif type == "average_size":
            eval = self.average_size_fitness
        else:
            raise ValueError("Invalid evaluation type")

        x = list(eval.keys())
        best = [eval[i].best for i in x]
        worst = [eval[i].worst for i in x]
        average = [eval[i].average for i in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, best, label='Best Fitness')
        plt.plot(x, worst, label='Worst Fitness')
        plt.plot(x, average, label='Average Fitness')

        type_to_ylabel = {
            "room_count": "Room Count",
            "average_distance": "Average Distance (m)",
            "average_size": "Average Room Size (m^2)"
        }

        type_to_title = {
            "room_count": "Room Count",
            "average_distance": "Average Distance",
            "average_size": "Average Room Size"
        }

        metric = type_to_title.get(EVALUATION_METHOD.value, "Unknown Metric")
        ylabel = type_to_ylabel.get(type, "Unknown Type")
        title = type_to_title.get(type, "Unknown Type")

        info_text = (
            f"Population size: {self.population_size}\n"
            f"Crossover rate: {self.crossover_rate}\n"
            f"Mutation rate: {self.mutation_rate}\n"
            f"Mutation points: {self.mutation_points}"
        )

        if not IS_MULTI_OBJECTIVE:
            info_text = f"Metric: {metric}\n" + info_text
            
        plt.text(0.01, 0.02, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.title(f"{title} Fitness Evaluation Over Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to file with timestamp
        folder = "fig/evaluation" if folder is None else folder
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{type}_{timestamp}" if filename is None else filename
        plt.savefig(f"{folder}/{filename}.png")

        # plt.show()
        plt.close()

    def plot_objective_space(self):
        f1_vals = [genome.count_used_rooms() for genome in self.population]
        f2_vals = [genome.calculate_average_distance() for genome in self.population]

        plt.figure(figsize=(8, 6))
        plt.scatter(f1_vals, f2_vals, color='blue', edgecolor='k', s=60)

        info_text = (
            f"Population size: {self.population_size}"
        )
        plt.text(0.01, 0.02, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.xlabel('Room Count')
        plt.ylabel('Average Distance (m)')
        plt.title('Population Objective Space')
        plt.grid(True)
        plt.tight_layout()

        # Save to file with timestamp
        os.makedirs("fig/objective_space", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"fig/objective_space/{timestamp}.png")

        # plt.show()
    
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
            if random.random() < self.crossover_rate and not identical:
                child1 = self.crossover(p1.chromosome, p2.chromosome)
                child2 = self.crossover(p2.chromosome, p1.chromosome)
                next_population.append(Genome(child1))
                next_population.append(Genome(child2))
            else:
                next_population.extend([Genome(p1.chromosome), Genome(p2.chromosome)])

        # Mutation
        for genome in next_population:
            if random.random() < self.mutation_rate:
                genome.mutate()

        self.population = next_population
        self.eval()
        
        if EVALUATION_METHOD.value == "room_count":
            self.best_genome = min(self.population, key=Genome.count_used_rooms)
        elif EVALUATION_METHOD.value == "average_distance":
            self.best_genome = min(self.population, key=Genome.calculate_average_distance)
        elif EVALUATION_METHOD.value == "average_size":
            self.best_genome = max(self.population, key=Genome.calculate_average_size)

        self.generation += 1

    def run(self):
        for _ in range(self.max_generation):
            self.evolve()
            if self.generation > 0.95 * self.max_generation:
                self.export_population()