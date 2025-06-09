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
    def __init__(self, context: ProblemContext, population_size: int, max_generation: int):
        super().__init__(context, population_size, max_generation)

        self.fronts: List[List[Genome]] = [[]]

    def plot_objective_space(self, color_by_rank: bool = False, connect_by_rank: bool = False):
        f1_vals = [genome.calculate_average_distance() for genome in self.population]
        f2_vals = [genome.calculate_average_size() for genome in self.population]

        plt.figure(figsize=(8, 6))

        self.non_dominated_sorting()
        if color_by_rank:
            ranks = [genome.rank for genome in self.population]
            scatter = plt.scatter(f1_vals, f2_vals, c=ranks, cmap='viridis', edgecolor='k', s=60)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Pareto Rank')
        else:
            plt.scatter(f1_vals, f2_vals, color='blue', edgecolor='k', s=60)

        # Optional: connect points in the same rank using same line style/color
        if connect_by_rank:
            from collections import defaultdict
            fronts = defaultdict(list)
            for genome in self.population:
                fronts[genome.rank].append(genome)

            for genomes_in_front in fronts.values():
                sorted_front = sorted(genomes_in_front, key=Genome.calculate_average_distance)
                x = [g.calculate_average_distance() for g in sorted_front]
                y = [g.calculate_average_size() for g in sorted_front]
                plt.plot(x, y, linestyle='--', linewidth=1, color='black')

        # Add population size as a textbox
        plt.gca().text(
            0.02, 0.98,
            f'Population Size: {self.population_size}',
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
        )

        plt.xlabel('Average Distance (m)')
        plt.ylabel('Average Size (m^2)')
        plt.title('Population Objective Space' + (' (Color by Rank)' if color_by_rank else ''))
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("fig/objective_space", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"fig/objective_space/{timestamp}.png")
        plt.show()

    def non_dominated_sorting(self, population: List[Genome] = None):
        if population is None:
            population = self.population

        checker = NonDominatedSorting(population)
        self.fronts = checker.run()

    def assign_crowding_distance(self):
        for front in self.fronts:
            CrowdingDistance(front).assign()

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
                offspring.append(Genome(child1))
                offspring.append(Genome(child2))
            else:
                offspring.extend([Genome(p1.chromosome), Genome(p2.chromosome)])

        # Mutation
        for genome in offspring:
            if random.random() < MUTATION_RATE:
                genome.mutate()

        self.non_dominated_sorting(self.population + offspring)
        self.population = self.select_next_generation()
        self.non_dominated_sorting()

        self.eval()
        self.generation += 1

    def run(self):
        for _ in range(self.max_generation):
            self.evolve()
            self.export_population()
