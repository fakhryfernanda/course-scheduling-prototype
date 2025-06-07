from typing import Final

TOTAL_DURATION: int = 0

LOG_EVALUATION: Final[bool] = False
CHECK_COUNTER: Final[bool] = True
SLOTS_PER_DAY: Final[int] = 5

POPULATION_SIZE: Final[int] = 10
MAX_GENERATION: Final[int] = 100
CROSSOVER_RATE: Final[float] = 0.7
MUTATION_RATE: Final[float] = 0.1

SELECTION_METHOD: Final[str] = "tournament"
RANDOMIZE_CROSSOVER: Final[bool] = True
MUTATION_METHOD: Final[str] = "random_swap"