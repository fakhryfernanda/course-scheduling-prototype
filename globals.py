from typing import Final
from enums.evaluation_method import EvaluationMethod
from dataframes.curriculum import Curriculum
from dataframes.subject import Subject
from dataframes.room import Room

subjects = Subject("csv/subjects.csv")
curriculum = Curriculum("csv/curriculum.csv", subjects.df)
rooms = Room("csv/rooms.csv")
COORDINATES: Final[dict[int, tuple[float, float]]] = rooms.df.set_index('id')[['lat', 'long']].apply(tuple, axis=1).to_dict()
SIZES: Final[dict[int, int]] = rooms.df.set_index('id')['capacity'].to_dict()

TOTAL_DURATION: Final[int] = (curriculum.df["classes"] * curriculum.df["credits"]).sum()
PARALLEL_COUNTS: Final[tuple[int]] = tuple(curriculum.df['classes'])

SLOTS_PER_DAY: Final[int] = 5

POPULATION_SIZE: Final[int] = 100
MAX_GENERATION: Final[int] = 100
CROSSOVER_RATE: Final[float] = 0.7
MUTATION_RATE: Final[float] = 0.2
MUTATION_POINTS: Final[int] = 5

EVALUATION_METHOD = EvaluationMethod.AVERAGE_SIZE
SELECTION_METHOD: Final[str] = "tournament"
CROSSOVER_METHOD: Final[str] = "column_based"
MUTATION_METHOD: Final[str] = "random_swap"