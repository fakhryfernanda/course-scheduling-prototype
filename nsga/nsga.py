import os
import random
import numpy as np
import matplotlib.pyplot as plt
from globals import *
from typing import List
from datetime import datetime
from ga.genome import Genome
from ga.genetic_algorithm import ProblemContext, GeneticAlgorithm
from nsga.non_dominated_sorting import NonDominatedSorting
from nsga.crowding_distance import CrowdingDistance

class NSGA2(GeneticAlgorithm):
    def __init__(self, context: ProblemContext, population_size: int, max_generation: int, crossover_rate: float, mutation_rate: float, mutation_points: int, seed: List[np.ndarray] = None):
        super().__init__(context, population_size, max_generation, crossover_rate, mutation_rate, mutation_points, seed)

        self.fronts: List[List[Genome]] = [[]]

    def plot_objective_space(
            self, 
            population: List[Genome] = None, 
            folder: str = None, 
            filename: str = None,
            color_by_rank: bool = False, 
            connect_by_rank: bool = False
        ):

        if population is None:
            population = self.population

        f1_vals = [genome.calculate_average_distance() for genome in population]
        f2_vals = [genome.calculate_average_size() for genome in population]

        fig, ax = plt.subplots(figsize=(8, 6))

        self.non_dominated_sorting()
        if color_by_rank:
            ranks = [genome.rank for genome in population]
            scatter = ax.scatter(f1_vals, f2_vals, c=ranks, cmap='viridis', edgecolor='k', s=60)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Pareto Rank')
        else:
            ax.scatter(f1_vals, f2_vals, color='blue', edgecolor='k', s=60)

        if connect_by_rank:
            from collections import defaultdict
            fronts = defaultdict(list)
            for genome in population:
                fronts[genome.rank].append(genome)

            for genomes_in_front in fronts.values():
                sorted_front = sorted(genomes_in_front, key=Genome.calculate_average_distance)
                x = [g.calculate_average_distance() for g in sorted_front]
                y = [g.calculate_average_size() for g in sorted_front]
                ax.plot(x, y, linestyle='--', linewidth=1, color='black')

        info_text = (
            f"Population size: {self.population_size}\n"
            f"Max generation: {self.max_generation}\n"
            f"Pareto front: {len(self.fronts[0])}\n"
            f"Crossover rate: {self.crossover_rate}\n"
            f"Mutation rate: {self.mutation_rate}\n"
            f"Mutation points: {self.mutation_points}"
        )

        # Place the info text outside the axes area
        fig.text(0.98, 0.02, info_text,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        ax.set_xlabel('Average Distance (m)')
        ax.set_ylabel('Average Size (m^2)')
        ax.set_title('Population Objective Space' + (' (Color by Rank)' if color_by_rank else ''))
        ax.grid(True)
        fig.tight_layout()

        folder = "fig/objective_space" if folder is None else folder
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = timestamp if filename is None else filename
        fig.savefig(f"{folder}/{filename}.png")
        # plt.show()
        plt.close()

    def non_dominated_sorting(self, population: List[Genome] = None):
        if population is None:
            population = self.population

        checker = NonDominatedSorting(population)
        self.fronts = checker.run()

    def assign_crowding_distance(self):
        for front in self.fronts:
            CrowdingDistance(front).assign()

    def deduplicate_population(self, population: List[Genome]):
        unique_population = []
        seen = set()

        for genome in population:
            key = tuple(genome.chromosome.flatten())
            if key not in seen:
                seen.add(key)
                unique_population.append(genome)

        return unique_population

    def select_next_generation(self) -> List[Genome]:
        self.assign_crowding_distance()

        next_population = []
        for front in self.fronts:
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                sorted_front = sorted(front, key=lambda g: g.crowding_distance, reverse=True)
                remaining = self.population_size - len(next_population)
                next_population.extend(sorted_front[:remaining])
                break

        return next_population
    
    def evolve(self):
        offspring = []

        # Selection
        parents = self.population

        # Crossover
        np.random.shuffle(parents)
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1]

            identical = np.array_equal(p1.chromosome, p2.chromosome)
            if random.random() < CROSSOVER_RATE and not identical:
                child1 = self.crossover(p1.chromosome, p2.chromosome)
                child2 = self.crossover(p2.chromosome, p1.chromosome)
                offspring.append(Genome(child1, self.config))
                offspring.append(Genome(child2, self.config))
            else:
                offspring.extend([Genome(p1.chromosome, self.config), Genome(p2.chromosome, self.config)])

        # Mutation
        for genome in offspring:
            if random.random() < self.mutation_rate:
                genome.mutate()

        combined = self.population + offspring
        combined = self.deduplicate_population(combined)
        if len(combined) < self.population_size:
            raise ValueError("Population size is not enough after deduplication")

        self.non_dominated_sorting(combined)
        self.population = self.select_next_generation()
        self.non_dominated_sorting()

        self.eval()
        self.generation += 1

    def run(self):
        for _ in range(self.max_generation):
            self.evolve()