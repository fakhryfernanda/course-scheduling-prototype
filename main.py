import random
import numpy as np
from utils import io
from globals import *
from dataframes.subject import Subject
from dataframes.curriculum import Curriculum
from ga.genetic_algorithm import GeneticAlgorithm, ProblemContext
from nsga.nsga import NSGA2

if __name__ == '__main__':
    subjects = Subject("csv/subjects.csv")
    curriculum = Curriculum("csv/curriculum.csv", subjects.df)

    context = ProblemContext(
        curriculum=curriculum,
        time_slot_indices=list(range(15)),
        room_indices=list(range(16))
    )

    crossover_rates = np.linspace(0.7, 0.9, 20)
    mutation_rates = np.linspace(0.1, 0.3, 20)

    all_combinations = list(product(crossover_rates, mutation_rates))
    all_combinations = [(round(c, 2), round(m, 2)) for c, m in all_combinations]
    params = random.sample(all_combinations, k=100)
    seed = io.import_all_txt_arrays("seed")

    for i, param in enumerate(params):
        crossover_rate, mutation_rate = param
        nsga = NSGA2(
            context=context,
            population_size=100,
            max_generation=100,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_points=MUTATION_POINTS,
            seed=seed
        )

        nsga.run()
        nsga.plot_objective_space(connect_by_rank=True, folder=f"simulation/run_{i+1}/objective_space", filename="after")

        nsga.plot_evaluation(type="average_distance", folder=f"simulation/run_{i+1}/evaluation", filename="average_distance")
        nsga.plot_evaluation(type="average_size", folder=f"simulation/run_{i+1}/evaluation", filename="average_size")

        pareto_front = nsga.fronts[0]
        nsga.export_population(pareto_front, folder=f"simulation/run_{i+1}/pareto_front")

