import random
from typing import List
from ga.genome import Genome

class ParentSelection:
    def __init__(self, method: str):
        self.method = method

    def run(self, population: List[Genome]) -> List[Genome]:
        if self.method == "tournament":
            return self._tournament_selection(population)
        elif self.method == "no_selection":
            return population
        else:
            raise ValueError(f"Selection method '{self.method}' is not supported.")

    def _tournament_selection(self, population: List[Genome]) -> List[Genome]:
        selected = []
        while len(selected) < len(population):
            competitors = random.sample(population, 2)
            winner = min(competitors, key=lambda g: g.count_used_rooms())
            selected.append(winner)
        return selected
