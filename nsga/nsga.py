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

    def plot_objective_space(self, color_by_rank: bool = False, connect_by_rank: bool = False):
        f1_vals = [genome.count_used_rooms() for genome in self.population]
        f2_vals = [genome.calculate_average_distance() for genome in self.population]

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
                sorted_front = sorted(genomes_in_front, key=lambda g: g.count_used_rooms())
                x = [g.count_used_rooms() for g in sorted_front]
                y = [g.calculate_average_distance() for g in sorted_front]
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

        plt.xlabel('Room Count')
        plt.ylabel('Average Distance (m)')
        plt.title('Population Objective Space' + (' (Color by Rank)' if color_by_rank else ''))
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("fig/objective_space", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"fig/objective_space/{timestamp}.png")
        plt.show()

    def non_dominated_sorting(self):
        checker = NonDominatedSorting(self.population)
        self.fronts = checker.run()