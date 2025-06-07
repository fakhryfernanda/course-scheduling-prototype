from typing import Final
from dataframes.curriculum import Curriculum
from dataframes.subject import Subject

subjects = Subject("csv/subjects.csv")
curriculum = Curriculum("csv/curriculum.csv", subjects.df)

TOTAL_DURATION: Final[int] = (curriculum.df["classes"] * curriculum.df["credits"]).sum()

SLOTS_PER_DAY: Final[int] = 5

POPULATION_SIZE: Final[int] = 10
MAX_GENERATION: Final[int] = 100
CROSSOVER_RATE: Final[float] = 0.7
MUTATION_RATE: Final[float] = 0.1

SELECTION_METHOD: Final[str] = "tournament"
CROSSOVER_METHOD: Final[str] = "row_based"
MUTATION_METHOD: Final[str] = "random_swap"