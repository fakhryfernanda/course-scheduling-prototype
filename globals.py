from dataclasses import dataclass
from typing import Final, Dict, Tuple
from pathlib import Path

from enums.evaluation_method import EvaluationMethod
from dataframes.subject import Subject
from dataframes.curriculum import Curriculum
from dataframes.room import Room


@dataclass
class Configuration:
    coordinates: Dict[int, Tuple[float, float]]
    sizes: Dict[int, int]
    total_duration: int
    parallel_counts: Tuple[int, ...]


def load_config(
    subjects_csv: str = "csv/subjects.csv",
    curriculum_csv: str = "csv/curriculum.csv",
    rooms_csv: str = "csv/rooms.csv",
) -> Configuration:
    """Load scheduling configuration from CSV files."""
    base_path = Path(__file__).resolve().parent

    subjects_path = base_path / subjects_csv
    curriculum_path = base_path / curriculum_csv
    rooms_path = base_path / rooms_csv

    subjects = Subject(subjects_path)
    curriculum = Curriculum(curriculum_path, subjects.df)
    rooms = Room(rooms_path)

    coordinates = (
        rooms.df.set_index("id")[["lat", "long"]]
        .apply(tuple, axis=1)
        .to_dict()
    )
    sizes = rooms.df.set_index("id")["size"].to_dict()
    total_duration = int(
        (curriculum.df["classes"] * curriculum.df["credits"]).sum()
    )
    parallel_counts = tuple(curriculum.df["classes"])

    return Configuration(
        coordinates=coordinates,
        sizes=sizes,
        total_duration=total_duration,
        parallel_counts=parallel_counts,
    )


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
