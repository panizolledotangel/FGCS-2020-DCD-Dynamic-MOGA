import random
import gc
from typing import List, Tuple

from deap import base
from deap import tools

import numpy as np
import pygmo

from sources.gas import creator
from sources.gas.auxiliary_funtions import type_crossover, count_types
from sources.gas.nsga2_config import NSGAIIConfig


class NSGAIISkeleton:

    def __init__(self, population: list, toolbox: base.Toolbox, config: NSGAIIConfig, ref_point: Tuple[float, float]):
        self.population = population
        self.toolbox = toolbox
        self.ref_point = ref_point
        self.population_size = len(population)
        self.ind_size = len(self.population[0])
        self.generations_without_change = 0
        self.config = config

    def start(self):
        # copy of the population
        pop = list(self.population)

        statistics = []
        generation = 0
        convergence = False

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = tools.selNSGA2(pop, self.config.population_size)

        # Log the pareto front
        pareto = tools.sortNondominated(pop, self.config.population_size, first_front_only=True)
        hv = pygmo.hypervolume([[i.fitness.values[0], i.fitness.values[1]] for i in pareto[0]])
        statistics.append(hv.compute(ref_point=self.ref_point))

        # Begin the generational process
        while not convergence and generation < self.config.number_generations:
            # Vary the population
            offspring = self.toolbox.select(pop, self.config.population_size)
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # crossover and mutation
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.config.crossover_rate:
                    self.toolbox.mate(ind1, ind2)

                    new_type = type_crossover(ind1, ind2)
                    ind1.i_type = new_type
                    ind2.i_type = new_type

                if random.random() <= self.config.mutation_rate:
                    self.toolbox.mutate(ind1)
                    self.toolbox.mutate(ind2)

                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = self.toolbox.dominance(pop + offspring, self.config.population_size)

            # Log the pareto front hypervolume
            pareto_new = tools.sortNondominated(pop, self.config.population_size, first_front_only=True)
            hv = pygmo.hypervolume([[i.fitness.values[0], i.fitness.values[1]] for i in pareto_new[0]])
            statistics.append(hv.compute(ref_point=self.ref_point))

            # check convergence
            # convergence = self._check_convergence(pareto[0], pareto_new[0])
            convergence = self._check_convergence_hv(statistics)
            pareto = pareto_new
            generation += 1

            gc.collect()

        return pop, pareto[0], statistics

    def _check_convergence(self, old_pareto: List[creator.Individual], new_pareto: List[creator.Individual]) -> bool:
        """
        Check if the algorithm have convergence checking if the pareto have not changed
        """

        # if the length is different the pareto have changed
        if len(old_pareto) != len(new_pareto):
            self.generations_without_change = 0
        else:
            # if both paretos have the same length we must check if the individuals are the same
            equal = True

            new_i = 0
            while equal and new_i < len(new_pareto):
                eq_found = False
                old_i = 0

                while not eq_found and old_i < len(old_pareto):
                    if np.array_equal(old_pareto[old_i], new_pareto[new_i]):
                        # if two individuals are then same, then the equality have been found
                        eq_found = True
                    old_i += 1

                # if in one case an identical individual have not been found then both paretos are different and the
                # search finishes
                equal = eq_found
                new_i += 1

            if equal:
                self.generations_without_change += 1
            else:
                self.generations_without_change = 0

        return self.generations_without_change > self.config.generations_to_converge

    def _check_convergence_hv(self, statistics: List[float]) -> bool:
        """
        Check if the algorithm have convergence measuring the variation in hypervolume
        :param statistics: list with hypervolume of the pareto fronts of the generations the algorithm have done
        :return: true if the algorithm have converged, false otherwise
        """
        convergence = False
        if len(statistics) > self.config.generations_to_converge:
            last_hypervolume = statistics[-self.config.generations_to_converge]
            actual_hypervolume = statistics[-1]

            convergence = abs(actual_hypervolume - last_hypervolume) < self.config.threshold_converge

        return convergence