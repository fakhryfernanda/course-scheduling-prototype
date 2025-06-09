import random
from typing import Final
from itertools import product
from enums.evaluation_method import EvaluationMethod
from dataframes.curriculum import Curriculum
from dataframes.subject import Subject
from dataframes.room import Room

# Define ranges
crossover_rates = [round(0.6 + 0.03 * i, 2) for i in range(11)]
mutation_rates = [round(0.01 + 0.03 * i, 2) for i in range(9)]
mutation_points = list(range(1, 6))

# Generate all combinations
all_combinations = list(product(crossover_rates, mutation_rates, mutation_points))

# Pick one random index
idx = random.randint(0, len(all_combinations) - 1)


subjects = Subject("csv/subjects.csv")
curriculum = Curriculum("csv/curriculum.csv", subjects.df)
rooms = Room("csv/rooms.csv")
COORDINATES: Final[dict[int, tuple[float, float]]] = rooms.df.set_index('id')[['lat', 'long']].apply(tuple, axis=1).to_dict()
SIZES: Final[dict[int, int]] = rooms.df.set_index('id')['size'].to_dict()

TOTAL_DURATION: Final[int] = (curriculum.df["classes"] * curriculum.df["credits"]).sum()
PARALLEL_COUNTS: Final[tuple[int]] = tuple(curriculum.df['classes'])

SLOTS_PER_DAY: Final[int] = 5

POPULATION_SIZE: Final[int] = 100
MAX_GENERATION: Final[int] = 100

CROSSOVER_RATE: Final[float] = 0.7
MUTATION_RATE: Final[float] = 0.2
MUTATION_POINTS: Final[int] = 5

IS_MULTI_OBJECTIVE: Final[bool] = True
EVALUATION_METHOD = EvaluationMethod.AVERAGE_SIZE
SELECTION_METHOD: Final[str] = "tournament"
CROSSOVER_METHOD: Final[str] = "column_based"
MUTATION_METHOD: Final[str] = "random_swap"