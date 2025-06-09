import random
from typing import List
from ga.genome import Genome
from globals import EVALUATION_METHOD
from enums.evaluation_method import EvaluationMethod

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
        if EVALUATION_METHOD == EvaluationMethod.ROOM_COUNT:
            metric = lambda g: g.count_used_rooms()
        elif EVALUATION_METHOD == EvaluationMethod.AVERAGE_DISTANCE:
            metric = lambda g: g.calculate_average_distance()
        else:
            raise ValueError(f"Unknown selection criteria: {EVALUATION_METHOD}")

        selected = []
        while len(selected) < len(population):
            competitors = random.sample(population, 2)
            winner = min(competitors, key=metric)
            selected.append(winner)

        return selected
