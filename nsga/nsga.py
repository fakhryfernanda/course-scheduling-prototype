import os
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
from ga.genome import Genome
from ga.genetic_algorithm import ProblemContext, GeneticAlgorithm
from nsga.non_dominated_sorting import NonDominatedSorting

class NSGA2(GeneticAlgorithm):
    def __init__(self, context: ProblemContext, population_size: int):
        super().__init__(context, population_size)

        self.fronts: List[List[Genome]] = [[]]

    def plot_objective_space(self):
        f1_vals = [genome.count_used_rooms() for genome in self.population]
        f2_vals = [genome.calculate_average_distance() for genome in self.population]

        plt.figure(figsize=(8, 6))
        plt.scatter(f1_vals, f2_vals, color='blue', edgecolor='k', s=60)

        plt.xlabel('Room Count')
        plt.ylabel('Average Distance (m)')
        plt.title('Population Objective Space')
        plt.grid(True)
        plt.tight_layout()

        # Save to file with timestamp
        os.makedirs("fig/objective_space", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"fig/objective_space/{timestamp}.png")

        plt.show()

    def non_dominated_sorting(self):
        checker = NonDominatedSorting(self.population)
        self.fronts = checker.run()