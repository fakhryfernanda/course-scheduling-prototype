import random
from typing import List
from ga.genome import Genome
from globals import EVALUATION_METHOD
from enums.evaluation_method import EvaluationMethod

class ParentSelection:
    def __init__(self, method: str):
        self.method = method
        self.evaluation_method = EVALUATION_METHOD

    def run(self, population: List[Genome]) -> List[Genome]:
        if self.method == "tournament":
            return self._tournament_selection(population)
        elif self.method == "no_selection":
            return population
        else:
            raise ValueError(f"Selection method '{self.method}' is not supported.")

    def _tournament_selection(self, population: List[Genome]) -> List[Genome]:
        if self.evaluation_method == EvaluationMethod.ROOM_COUNT:
            metric = lambda g: g.count_used_rooms()
        elif self.evaluation_method == EvaluationMethod.AVERAGE_DISTANCE:
            metric = lambda g: g.calculate_average_distance()
        elif self.evaluation_method == EvaluationMethod.AVERAGE_SIZE:
            metric = lambda g: g.calculate_average_size()
        else:
            raise ValueError(f"Unknown selection criteria: {self.evaluation_method}")

        selected = []
        while len(selected) < len(population):
            competitors = random.sample(population, 2)
            if self.evaluation_method == EvaluationMethod.AVERAGE_SIZE:
                winner = max(competitors, key=metric)
            else:
                winner = min(competitors, key=metric)
            selected.append(winner)

        return selected
