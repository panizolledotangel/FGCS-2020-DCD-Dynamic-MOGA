from deap import base
from deap import creator

IMMIGRANT = 0
IMMIGRANT_IMMIGRANT = 1
RANDOM = 2
RANDOM_RANDOM = 3
IMMIGRANT_RANDOM = 4

creator.create("FitnessODF_CSCORE", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessODF_CSCORE, i_type=IMMIGRANT, n_repro=0)

ref_point = [1.1, 1.1]
