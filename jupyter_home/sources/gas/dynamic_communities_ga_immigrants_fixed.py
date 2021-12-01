import math

import numpy as np
import gc
from deap import tools

import sources.gas.auxiliary_funtions as auxf
from sources.gas import RANDOM, IMMIGRANT
from sources.gas.dynamic_communities_ga_standard import DynamicCommunitiesGAStandard
from sources.gas.dynamic_ga_immigrants_config import DynamicGaImmigrantsConfiguration
from sources.gas.nsga2_skeleton import NSGAIISkeleton
from sources.gloaders.loader_interface import LoaderInterface


class DynamicCommunitiesGAImmigrantsFixed(DynamicCommunitiesGAStandard):

    def __init__(self, dataset: LoaderInterface, config: DynamicGaImmigrantsConfiguration):
        super().__init__(dataset, config)

    def make_dict(self):
        return self.config.make_dict()

    def find_communities(self):

        snp_size = len(self.dataset.snapshots)
        snapshot_members = None
        snapshot_pareto = [None] * snp_size

        best_pop = None
        for i in range(snp_size):
            print("working on snapshot {0}...".format(i))
            self.config.set_snapshot(i)

            if i is 0:
                actual_g = self.dataset.snapshots[i]
                # Initialize GA
                toolbox = self.config.ga_configs.make_toolbox(actual_g)
                pop_initial = [toolbox.individual(RANDOM) for i in range(self.config.get_ga_config().population_size)]

                # Evaluate the individuals
                fitnesses = toolbox.map(toolbox.evaluate, pop_initial)
                for ind, fit in zip(pop_initial, fitnesses):
                    ind.fitness.values = fit

                ga = NSGAIISkeleton(pop_initial, toolbox, self.config.get_ga_config(), auxf.get_ref_point(actual_g))
            else:
                actual_g = self.dataset.snapshots[i]
                previous_g = self.dataset.snapshots[i-1]
                previous_sol = snapshot_members

                # Initialize GA
                toolbox = self.config.make_toolbox(actual_snapshot=actual_g, previous_snapshot=previous_g,
                                                   previous_solution=previous_sol)

                pop_initial, _, _, _ = self._select_immigrants(toolbox, best_pop)
                ga = self._make_NSGAII(pop_initial, toolbox, auxf.get_ref_point(actual_g))

            # evolve population
            best_pop, pareto, _ = ga.start()

            # save solution
            snapshot_members = auxf.decode(self._select_solution(pareto, actual_g))
            snapshot_pareto[i] = pareto
            gc.collect()

        r_data = {}
        return r_data, [], snapshot_pareto

    def _select_immigrants(self, tbox, past_population):
        num_elite_immigrants = int(math.ceil((1 - self.config.get_rate_random_immigrants()) *
                                             self.config.get_ga_config().population_size))

        num_random_immigrants = self.config.get_ga_config().population_size - num_elite_immigrants

        assert num_elite_immigrants + num_random_immigrants == self.config.get_ga_config().population_size, \
            "new population exceeds population size {0}".format(num_elite_immigrants + num_random_immigrants)

        if num_elite_immigrants > 0:
            elite_immigrants = list(tbox.map(tbox.clone, tools.selBest(past_population, num_elite_immigrants)))
            repaired_output = [tbox.repair(individual=x) for x in elite_immigrants]

            elite_immigrants, n_gen_repaired = zip(*repaired_output)
            elite_immigrants = list(elite_immigrants)

            invalid_ind = [ind for ind in elite_immigrants if not ind.fitness.valid]
            fitnesses = tbox.map(tbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.i_type = IMMIGRANT
        else:
            elite_immigrants = []
            n_gen_repaired = 0

        random_immigrants = [tbox.individual(RANDOM) for _ in range(num_random_immigrants)]
        fitnesses = tbox.map(tbox.evaluate, random_immigrants)
        for ind, fit in zip(random_immigrants, fitnesses):
            ind.fitness.values = fit

        immigrants = elite_immigrants + random_immigrants
        return immigrants, num_elite_immigrants, num_random_immigrants, n_gen_repaired
