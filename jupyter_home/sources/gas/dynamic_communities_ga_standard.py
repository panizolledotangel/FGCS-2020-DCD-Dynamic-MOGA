from typing import List, Tuple

import igraph
import gc
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
        snapshot_pareto = [None] * snp_size

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

            # evolve population
            ga = self._make_NSGAII(pop_initial, toolbox, auxf.get_ref_point(g))
            _, pareto, _ = ga.start()

            # save statistics
            snapshot_pareto[i] = pareto

            # clean memory
            ga = None
            gc.collect()

        r_data = {}
        return r_data, [], snapshot_pareto

    def _make_NSGAII(self, pop_initial: List[creator.Individual], toolbox: base.Toolbox, ref_point: Tuple[float, float]) -> NSGAIISkeleton:

        config = self.config.get_ga_config()
        if config.__class__.__name__ == 'NSGAIIConfig':
            nsgaii = NSGAIISkeleton(pop_initial, toolbox, config, ref_point)
        else:
            raise RuntimeError("Unknown settings class {0}".format(config.__class__.__name__))

        return nsgaii
