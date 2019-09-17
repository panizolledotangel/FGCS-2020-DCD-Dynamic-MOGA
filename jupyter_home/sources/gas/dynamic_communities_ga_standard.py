from typing import List, Tuple

import igraph
import numpy as np
from deap import base

import sources.gas.auxiliary_funtions as auxf
from sources.gas import creator, RANDOM

from sources.gas.nsga2_skeleton import NSGAIISkeleton

from sources.gas.dynamic_ga_configuration import DynamicGaConfiguration
from sources.gloaders.loader_interface import LoaderInterface


class DynamicCommunitiesGAStandard:

    @classmethod
    def _select_solution(cls, pareto: List[creator.Individual], g: igraph.Graph):
        solution = None
        max_mod = float('-inf')

        for ind in pareto:
            mod = auxf.modularity_individual(ind, g)

            if mod > max_mod:
                solution = ind
                max_mod = mod
        return solution

    def __init__(self, dataset: LoaderInterface, config: DynamicGaConfiguration):
        self.dataset = dataset
        self.config = config

    def make_dict(self):
        return self.config.make_dict()

    def find_communities(self):
        snp_size = len(self.dataset.snapshots)
        snapshot_members = [None] * snp_size
        snapshot_generations = [None] * snp_size
        snapshot_whole_population = [None] * snp_size
        snapshot_pareto = [None] * snp_size

        generations_taken = [0]*snp_size

        # population types
        snapshot_population_types = []

        for i in range(snp_size):
            self.config.set_snapshot(i)
            g = self.dataset.snapshots[i]

            # Initialize GA
            toolbox = self.config.make_toolbox(snapshot=g)

            pop_initial = [toolbox.individual(RANDOM) for i in range(self.config.get_ga_config().population_size)]

            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, pop_initial)
            for ind, fit in zip(pop_initial, fitnesses):
                ind.fitness.values = fit

            print("working on snapshot {0}...".format(i))

            # Matrix for populations
            population_size = self.config.get_ga_config().population_size
            individual_size = len(pop_initial[0])
            pop_matrix = np.zeros((2, population_size, individual_size), dtype=int)

            # Log initial population
            for index, ind in enumerate(pop_initial):
                pop_matrix[0, index, :] = np.array(ind, dtype=int)

            # evolve population
            ga = self._make_NSGAII(pop_initial, toolbox, auxf.get_ref_point(g))
            best_pop, pareto, statistics = ga.start()

            # Log final population
            for index, ind in enumerate(best_pop):
                pop_matrix[1, index, :] = np.array(ind, dtype=int)

            # save statistics
            snapshot_generations[i] = statistics
            generations_taken[i] = len(statistics)

            # save solution
            snapshot_members[i] = auxf.decode(self._select_solution(pareto, g))
            snapshot_pareto[i] = pareto

            # save whole population
            snapshot_whole_population[i] = pop_matrix

            # save population types
            snapshot_population_types.append(ga.population_types)

        r_data = {
            "snapshot_members": snapshot_members,
            "generations_taken": generations_taken,
            "population_types": snapshot_population_types
        }
        return r_data, snapshot_generations, snapshot_whole_population, snapshot_pareto

    def _make_NSGAII(self, pop_initial: List[creator.Individual], toolbox: base.Toolbox, ref_point: Tuple[float, float]) -> NSGAIISkeleton:

        config = self.config.get_ga_config()
        if config.__class__.__name__ == 'NSGAIIConfig':
            nsgaii = NSGAIISkeleton(pop_initial, toolbox, config, ref_point)
        else:
            raise RuntimeError("Unknown settings class {0}".format(config.__class__.__name__))

        return nsgaii
