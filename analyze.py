import random
import numpy as np
from utils import io
from globals import *
from globals import load_config
from dataframes.subject import Subject
from dataframes.curriculum import Curriculum
from ga.genetic_algorithm import GeneticAlgorithm, ProblemContext
from nsga.nsga import NSGA2

if __name__ == '__main__':
    subjects = Subject("csv/subjects.csv")
    curriculum = Curriculum("csv/curriculum.csv", subjects.df)
    config = load_config()

    context = ProblemContext(
        curriculum=curriculum,
        time_slot_indices=list(range(15)),
        room_indices=list(range(16)),
        config=config,
    )

    seed = io.import_all_txt_arrays("all_pareto_fronts")

    nsga = NSGA2(
        context=context,
        population_size=1026,
        max_generation=10,
        crossover_rate=0.7,
        mutation_rate=0.1,
        mutation_points=MUTATION_POINTS,
        seed=seed
    )

    nsga.plot_objective_space(connect_by_rank=True)