from functools import partial, singledispatch
from typing import Dict
import importlib

import igraph
import pickle
from bson import Binary
from deap import creator
from deap.base import Toolbox
from deap.tools import cxTwoPoint, selTournamentDCD, selNSGA2

from sources.gas.auxiliary_funtions import create_individual, mutate_individual, evaluate_individual

"""
type overloaded function to get the module and the name of functions and partials, for DB storing purposes
"""


@singledispatch
def make_functions_str(f):
    return "{0}.{1}".format(f.__module__, f.__name__)


@make_functions_str.register(partial)
def _(f):
    return "{0}.{1}".format(f.func.__module__, f.func.__name__)


""""""


class NSGAIIConfig:

    @classmethod
    def load_from_dict(cls, d: Dict):
        ga_config = NSGAIIConfig(
            new_individual_func=pickle.loads(d['new_individual_binary']),
            crossover_func=pickle.loads(d['crossover_binary']),
            selection_func=pickle.loads(d['selection_binary']),
            mutate_func=pickle.loads(d['mutate_binary']),
            evaluate_func=pickle.loads(d['evaluate_binary']),
            dominance_func=pickle.loads(d['dominance_binary']),
            crossover_rate=float(d['crossover_rate']),
            mutation_rate=float(d['mutation_rate']),
            number_generations=int(d['max_num_generations']),
            population_size=int(d['population_size']),
            offspring_size=int(d['offspring_size']),
            generations_to_converge=int(d['generations_to_converge']),
            threshold_converge=float(d['threshold_converge'])
        )
        return ga_config

    def __init__(self, new_individual_func=create_individual, crossover_func=cxTwoPoint, selection_func=selTournamentDCD,
                 dominance_func=selNSGA2, mutate_func=mutate_individual, evaluate_func=evaluate_individual,
                 crossover_rate=1.0, mutation_rate=0.1, number_generations=300, population_size=300, offspring_size=300,
                 generations_to_converge=20, threshold_converge=1.0):

        self.new_individual_func = new_individual_func
        self.crossover_func = crossover_func
        self.selection_func = selection_func
        self.dominance_func = dominance_func
        self.mutate_func = mutate_func
        self.evaluate_func = evaluate_func

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.number_generations = number_generations
        self.offspring_size = offspring_size
        self.population_size = population_size
        self.generations_to_converge = generations_to_converge
        self.threshold_converge = float(threshold_converge)

    def make_dict(self):
        d = {
            "new_individual_func": make_functions_str(self.new_individual_func),
            "crossover_func": make_functions_str(self.crossover_func),
            "selection_func": make_functions_str(self.selection_func),
            "mutate_func": make_functions_str(self.mutate_func),
            "evaluate_func": make_functions_str(self.evaluate_func),
            "dominance_func": make_functions_str(self.dominance_func),

            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "population_size": self.population_size,
            "max_num_generations": self.number_generations,
            "offspring_size": self.offspring_size,
            "generations_to_converge": self.generations_to_converge,
            "threshold_converge": self.threshold_converge
        }
        return d

    def serialize(self):
        d = self.make_dict()
        d.update({
            "new_individual_binary": Binary(pickle.dumps(self.new_individual_func, protocol=2), subtype=128),
            "crossover_binary": Binary(pickle.dumps(self.crossover_func, protocol=2), subtype=128),
            "selection_binary": Binary(pickle.dumps(self.selection_func, protocol=2), subtype=128),
            "mutate_binary": Binary(pickle.dumps(self.mutate_func, protocol=2), subtype=128),
            "evaluate_binary": Binary(pickle.dumps(self.evaluate_func, protocol=2), subtype=128),
            "dominance_binary": Binary(pickle.dumps(self.dominance_func, protocol=2), subtype=128),
            "module": str(self.__class__.__name__)
        })
        return d

    def make_toolbox(self, g: igraph.Graph) -> Toolbox:
        toolbox = Toolbox()

        individual_size = g.vcount()
        toolbox.register("individual", self.new_individual_func, container=creator.Individual, graph=g,
                         n=individual_size)

        toolbox.register("mate", self.crossover_func)
        toolbox.register("mutate", self.mutate_func, graph=g, probability=self.mutation_rate)
        toolbox.register("select", self.selection_func)
        toolbox.register("dominance", self.dominance_func)
        toolbox.register("evaluate", self.evaluate_func, graph=g)
        return toolbox
