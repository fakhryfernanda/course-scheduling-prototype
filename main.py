from globals import *
from dataframes.subject import Subject
from dataframes.curriculum import Curriculum
from ga.genetic_algorithm import GeneticAlgorithm, ProblemContext

if __name__ == '__main__':
    subjects = Subject("csv/subjects.csv")
    curriculum = Curriculum("csv/curriculum.csv", subjects.df)

    context = ProblemContext(
        curriculum=curriculum,
        time_slot_indices=list(range(15)),
        room_indices=list(range(7))
    )

    ga = GeneticAlgorithm(
        context=context, 
        population_size=POPULATION_SIZE, 
    )
