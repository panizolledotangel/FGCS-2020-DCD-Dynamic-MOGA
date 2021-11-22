import math

import numpy as np
from deap import tools

import sources.gas.auxiliary_funtions as auxf
from sources.gas import RANDOM, IMMIGRANT
from sources.gas.dynamic_communities_ga_standard import DynamicCommunitiesGAStandard
from sources.gas.dcd_gas_immigrants_combine_reparators_config import DCDGasImmigrantsCombineReparatorsConfig
from sources.gas.nsga2_skeleton import NSGAIISkeleton
from sources.gloaders.loader_interface import LoaderInterface


class DCDGasImmigrantsCombineReparators(DynamicCommunitiesGAStandard):

    def __init__(self, dataset: LoaderInterface, config: DCDGasImmigrantsCombineReparatorsConfig):
        super().__init__(dataset, config)

    def make_dict(self):
        return self.config.make_dict()

    def find_communities(self):

        snp_size = len(self.dataset.snapshots)
        snapshot_members = [None] * snp_size
        snapshot_generations = [None] * snp_size
        snapshot_pareto = [None] * snp_size

        immigrants = [None] * snp_size
        repaired_list = [None] * snp_size

        generations_taken = [0]*snp_size

        # population types
        snapshot_population_types = []

        best_pop = None
        for i in range(snp_size):
            print("working on snapshot {0}...".format(i))
            self.config.set_snapshot(i)

            if i is 0:
                actual_g = self.dataset.snapshots[i]
                # Initialize GA
                toolbox = self.config.ga_configs.make_toolbox(actual_g)

                n_gen_repaired = [0] * self.config.get_ga_config().population_size
                n_elite = 0
                n_random = self.config.get_ga_config().population_size
                pop_initial = [toolbox.individual(RANDOM) for i in range(self.config.get_ga_config().population_size)]

                # Evaluate the individuals
                fitnesses = toolbox.map(toolbox.evaluate, pop_initial)
                for ind, fit in zip(pop_initial, fitnesses):
                    ind.fitness.values = fit

                ga = NSGAIISkeleton(pop_initial, toolbox, self.config.get_ga_config(), auxf.get_ref_point(actual_g))
            else:
                actual_g = self.dataset.snapshots[i]
                previous_g = self.dataset.snapshots[i-1]
                previous_sol = snapshot_members[i-1]

                # Initialize GA
                toolbox = self.config.make_toolbox(actual_snapshot=actual_g, previous_snapshot=previous_g,
                                                   previous_solution=previous_sol)

                pop_initial, n_elite, n_random, n_gen_repaired = self._select_immigrants(toolbox, best_pop)
                ga = self._make_NSGAII(pop_initial, toolbox, auxf.get_ref_point(actual_g))

            # evolve population
            best_pop, pareto, statistics = ga.start()

            # save statistics
            snapshot_generations[i] = statistics
            generations_taken[i] = len(statistics)

            # save immigrants
            immigrants[i] = [n_elite, n_random]
            repaired_list[i] = n_gen_repaired

            # save solution
            snapshot_members[i] = auxf.decode(self._select_solution(pareto, actual_g))
            snapshot_pareto[i] = pareto

            # save population types
            snapshot_population_types.append(ga.population_types)

        r_data = {
            "snapshot_members": snapshot_members,
            "generations_taken": generations_taken,
            "immigrants": immigrants,
            "repaired_list": repaired_list,
            "population_types": snapshot_population_types
        }
        return r_data, snapshot_generations, snapshot_pareto

    def _select_immigrants(self, tbox, past_population):
        num_elite_immigrants = int(math.ceil((1 - self.config.get_rate_random_immigrants()) *
                                             self.config.get_ga_config().population_size))

        num_random_immigrants = self.config.get_ga_config().population_size - num_elite_immigrants

        assert num_elite_immigrants + num_random_immigrants == self.config.get_ga_config().population_size, \
            "new population exceeds population size {0}".format(num_elite_immigrants + num_random_immigrants)

        if num_elite_immigrants > 0:
            elite_immigrants, _ = self.config.sel_function(past_population, num_elite_immigrants)
            elite_immigrants = list(tbox.map(tbox.clone, elite_immigrants))

            repaired_output = [tbox.repair_1(individual=x) for x in elite_immigrants]
            repaired_output.extend([tbox.repair_2(individual=x) for x in elite_immigrants])

            elite_immigrants, n_gen_repaired = zip(*repaired_output)

            fitnesses = tbox.map(tbox.evaluate, elite_immigrants)
            for ind, fit in zip(elite_immigrants, fitnesses):
                ind.fitness.values = fit
                ind.i_type = IMMIGRANT

            # select the best
            elite_immigrants = tbox.dominance(elite_immigrants, num_elite_immigrants)
            elite_immigrants = list(elite_immigrants)
        else:
            elite_immigrants = []
            n_gen_repaired = 0

        random_immigrants = [tbox.individual(RANDOM) for _ in range(num_random_immigrants)]
        fitnesses = tbox.map(tbox.evaluate, random_immigrants)
        for ind, fit in zip(random_immigrants, fitnesses):
            ind.fitness.values = fit

        immigrants = elite_immigrants + random_immigrants
        return immigrants, num_elite_immigrants, num_random_immigrants, n_gen_repaired[0:num_elite_immigrants]
