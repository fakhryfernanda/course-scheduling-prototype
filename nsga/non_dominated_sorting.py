import numpy as np
from typing import List
from ga.genome import Genome
from utils.helper import normalize_objectives

class NonDominatedSorting:
    def __init__(self, population: List[Genome]):
        self.population = population
        self.population_size = len(population)

    def reset_genomes(self):
        for genome in self.population:
            genome.reset_state()

    def run(self):
        self.reset_genomes()
        self.perform_domination_checks()
        return self.build_fronts()

    def perform_domination_checks(self):
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                g1 = self.population[i]
                g2 = self.population[j]
                self.domination_check(g1, g2)
                self.domination_check(g2, g1)

    def domination_check(self, g1: Genome, g2: Genome):
        g1_eval = g1.get_objectives()
        g2_eval = g2.get_objectives()

        maximize_mask = np.array([False, True])  # index 0: minimize, index 1: maximize
        g1_norm = normalize_objectives(g1_eval, maximize_mask)
        g2_norm = normalize_objectives(g2_eval, maximize_mask)

        no_worse = np.all(g1_norm <= g2_norm)
        strictly_better = np.any(g1_norm < g2_norm)

        if no_worse and strictly_better:
            g1.dominated_set.append(g2)
            g2.domination_count += 1

    def build_fronts(self):
        fronts = [[]]
        for genome in self.population:
            if genome.domination_count == 0:
                genome.rank = 0
                fronts[0].append(genome)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]
