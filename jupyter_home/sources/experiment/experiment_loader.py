from typing import List, Tuple
from datetime import datetime, timedelta
import pickle

import igraph
import numpy as np
from pytimeparse.timeparse import timeparse

import sources.mongo_connection.mongo_queries as db_queries
from sources.gloaders.loader_interface import LoaderInterface
from sources.gas.auxiliary_funtions import decode, modularity_individual, average_odf_and_community_score


class ExperimentLoader:

    @classmethod
    def _calculate_modularity(cls, members: List[List[int]], dataset_ob: LoaderInterface) -> np.array:
        modularity = [g.modularity(members[i]) for i, g in enumerate(dataset_ob.snapshots)]
        return np.array(modularity)

    def __init__(self, dataset_id: str, settings_id: str):
        self.dataset = db_queries.get_dataset(dataset_id)
        self.settings = db_queries.get_settings(settings_id)

        with db_queries.find_iteration(dataset_id, settings_id) as cursor:
            self.iteration_list = list(cursor)

    def get_population_types_matrix(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        max_generations = db_queries.max_generations(self.dataset['_id'], self.settings['_id'])
        return (self._make_pop_matrix(max_generations,  'immigrant'),
                self._make_pop_matrix(max_generations,  'immigrant_immigrant'),
                self._make_pop_matrix(max_generations,  'random'),
                self._make_pop_matrix(max_generations,  'random_random'),
                self._make_pop_matrix(max_generations,  'random_immigrant'))

    def get_whole_population_types_matrix(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        max_generations_snp = np.max(self.get_generations_taken_matrix(), axis=0).astype(int)

        return (self._make_whole_pop_matrix(max_generations_snp, 'immigrant'),
                self._make_whole_pop_matrix(max_generations_snp, 'immigrant_immigrant'),
                self._make_whole_pop_matrix(max_generations_snp, 'random'),
                self._make_whole_pop_matrix(max_generations_snp, 'random_random'),
                self._make_whole_pop_matrix(max_generations_snp, 'random_immigrant'))

    def get_solutions_fitness(self) -> np.array:
        """
        (nº interations, nº snapshot, 2)
        :return:
        """
        dataset_obj = db_queries.get_dataset_obj(self.dataset['_id'])

        fitness_matrix = np.zeros((len(self.iteration_list), self.dataset['snapshot_count'], 2))
        for n_it, it in enumerate(self.iteration_list):

            snp_members = it['execution_info']['snapshot_members']
            for n_snp in range(self.dataset['snapshot_count']):

                avg_odf, c_score = average_odf_and_community_score(dataset_obj.snapshots[n_snp], snp_members[n_snp])
                fitness_matrix[n_it, n_snp, 0] = avg_odf
                fitness_matrix[n_it, n_snp, 1] = c_score

        return fitness_matrix

    def get_modularity_matrix(self) -> np.array:
        """
        (nº iterations, nº snapshots)
        :return:
        """
        dataset_obj = db_queries.get_dataset_obj(self.dataset['_id'])

        modularity_matrix = np.zeros((len(self.iteration_list), self.dataset['snapshot_count']))
        for i, it in enumerate(self.iteration_list):
            modularity_matrix[i, :] = ExperimentLoader._calculate_modularity(it["execution_info"]['snapshot_members'],
                                                                             dataset_obj)
        return modularity_matrix

    def get_gen_repaired_matrix(self) -> np.array:
        """
        (nº iterations, n_snapshots, pop_size)
        :return:
        """
        n_gen_repaired_matrix = np.zeros((len(self.iteration_list), self.dataset['snapshot_count'], self.settings['ga_config']['population_size']))
        for n_it, it in enumerate(self.iteration_list):
            n_gen_repaired_matrix[n_it, :, :] = np.array(it['execution_info']['repaired_list'])

        return n_gen_repaired_matrix

    def get_gen_normalized_repaired_matrix(self) -> np.array:
        """
        (nº iterations, n_snapshots, pop_size)
        :return:
        """
        n_nodes = self.dataset['n_nodes']
        n_gen_repaired_matrix = np.zeros((len(self.iteration_list), self.dataset['snapshot_count'], self.settings['ga_config']['population_size']))

        for n_it, it in enumerate(self.iteration_list):
            for snp in range(self.dataset['snapshot_count']):
                n_gen_repaired_matrix[n_it, snp, :] = np.array(it['execution_info']['repaired_list'][snp]) / n_nodes[snp]

        return n_gen_repaired_matrix

    def get_times_matrix(self) -> np.array:
        """
        (nº iterations)
        :return:
        """
        times_matrix = np.zeros(len(self.iteration_list))
        for i, it in enumerate(self.iteration_list):
            duration_str = it['execution_info']['duration']
            times_matrix[i] = timeparse(duration_str)

        return times_matrix

    def get_generations_taken_matrix(self) -> np.array:
        """
        Returns a matrix with the generations taken by each execution of the algorithm
        :return:
        """
        generations_matrix = np.zeros((len(self.iteration_list), self.dataset['snapshot_count']))
        for i, it in enumerate(self.iteration_list):
            generations_matrix[i, :] = np.array(it['execution_info']['generations_taken'])

        return generations_matrix

    def get_hypervolume_matrix(self) -> np.array:
        """
        Returns the hypervolume of the paretos of the experiment for each iteration, snapshot and generation in a matrix
        of shape (nº iterations, nº snapshot, nº generation) the nº of generations is the maximum number of generations
        taken by any iteration, for those iterations sorter the vector is filled with the maximum value
        :return: a matrix of shape (nº iterations, nº snapshot, nº generation) with the hypervolume values
        """
        max_generations = db_queries.max_generations(self.dataset['_id'], self.settings['_id'])
        return self._make_hypervolume_matrix(max_generations)

    def get_join_hypervolume_padded_matrix(self, max_generations) -> np.array:
        """
        Returns the hypervolume of the paretos of the experiment for each iteration and generation in a matrix
        of shape (nº iterations, nº generation) the nº of generations is the maximum number of generations
        taken by any iteration, for those iterations sorter the vector is filled with the maximum value
        :return: a matrix of shape (nº iterations, nº snapshot, nº generation) with the hypervolume values
        """
        number_iterations = len(self.iteration_list)

        hv_matrix = np.zeros((number_iterations, np.sum(max_generations)))
        for n_it, iteration in enumerate(self.iteration_list):

            hv = np.zeros(np.sum(max_generations))
            s_index = 0
            for i, snp in enumerate(iteration['snapshots']):
                values_to_add = max_generations[i] - len(snp)

                e_index = s_index + max_generations[i]
                hv[s_index:e_index] = np.pad(np.array(snp), (0, values_to_add), 'maximum')

                s_index += max_generations[i]

            hv_matrix[n_it, :] = hv

        return hv_matrix

    def get_join_hypervolume_matrix(self) -> np.array:
        """
        Returns the hypervolume of the paretos of the experiment for each iteration and generation in a matrix
        of shape (nº iterations, nº generation) the nº of generations is the maximum number of generations
        taken by any iteration, for those iterations sorter the vector is filled with the maximum value
        :return: a matrix of shape (nº iterations, nº snapshot, nº generation) with the hypervolume values
        """
        number_iterations = len(self.iteration_list)
        max_generations_snp = np.max(self.get_generations_taken_matrix(), axis=0).astype(int)
        max_generations = int(np.sum(max_generations_snp))

        hv_matrix = np.zeros((number_iterations, max_generations))
        for n_it, iteration in enumerate(self.iteration_list):

            hv = np.zeros(max_generations)
            s_index = 0
            for i, snp in enumerate(iteration['snapshots']):
                values_to_add = max_generations_snp[i] - len(snp)

                e_index = s_index + max_generations_snp[i]
                hv[s_index:e_index] = np.pad(np.array(snp), (0, values_to_add), 'maximum')

                s_index += max_generations_snp[i]

            hv_matrix[n_it, :] = hv

        return hv_matrix

    def get_max_min_paretos(self)-> Tuple[List[np.array], List[np.array]]:
        """
        Return the pareto front of the solution with max and min hypervolume value for each snapshot
        :return: a tuple with the best and worse  pareto fronts
        """
        hv = np.zeros(len(self.iteration_list))
        for n_it, it in enumerate(self.iteration_list):
            hv[n_it] = np.sum(np.array([max(snp) for snp in it['snapshots']]))

        max_pareto_iter = int(np.argmax(hv))
        min_pareto_iter = int(np.argmin(hv))

        return self._extract_pareto_hv(max_pareto_iter), self._extract_pareto_hv(min_pareto_iter)

    def get_median_paretos_hv_mod(self)-> Tuple[List[np.array], List[np.array]]:
        """
        Return the pareto front of the solution with max and min hypervolume value for each snapshot
        :return: a tuple with the best and worse  pareto fronts
        """
        hv = np.zeros(len(self.iteration_list))
        for n_it, it in enumerate(self.iteration_list):
            hv[n_it] = np.sum(np.array([max(snp) for snp in it['snapshots']]))

        pareto_iter = int(np.argsort(hv)[len(hv)//2])
        return self._extract_pareto_hv(pareto_iter), self._extract_pareto_modularity(pareto_iter)

    def get_all_paretos_hv_sol(self)-> List[Tuple[List[np.array], np.array]]:
        """
        Return the pareto front of the solution with max and min hypervolume value for each snapshot
        :return: a tuple with the best and worse  pareto fronts
        """
        solutions_fitness = self.get_solutions_fitness()

        paretos = []
        for n_it in range(len(self.iteration_list)):
            act_pareto = self._extract_pareto_hv(n_it)
            act_sol_fitness = solutions_fitness[n_it, :, :]
            paretos.append((act_pareto, act_sol_fitness))

        return paretos

    def get_max_min_hv_solution(self) -> Tuple[List[List[int]], List[List[int]]]:
        """

        :return:
        """
        hv = np.zeros(len(self.iteration_list))
        for n_it, it in enumerate(self.iteration_list):
            hv[n_it] = np.sum(np.array([max(snp) for snp in it['snapshots']]))

        max_hv_iter = int(np.argmax(hv))
        min_hv_iter = int(np.argmin(hv))

        return self.iteration_list[max_hv_iter]['execution_info']['snapshot_members'], self.iteration_list[min_hv_iter]['execution_info']['snapshot_members']

    def get_median_hv_solution(self) -> List[List[int]]:
        """

        :return:
        """
        hv = np.zeros(len(self.iteration_list))
        for n_it, it in enumerate(self.iteration_list):
            hv[n_it] = np.sum(np.array([max(snp) for snp in it['snapshots']]))

        pareto_iter = int(np.argsort(hv)[len(hv) // 2])
        return self.iteration_list[pareto_iter]['execution_info']['snapshot_members']

    def get_mni_matrix(self) -> np.array:
        if self.dataset['ground_truth']:
            if 'memebers' in self.dataset:
                ground_truth = self.dataset['memebers']
            else:
                ground_truth = self.dataset['members']

            n_snp = self.dataset['snapshot_count']

            mni_matrix = np.zeros((len(self.iteration_list), n_snp))
            for i, it in enumerate(self.iteration_list):
                try:
                    it_solution = it["execution_info"]['snapshot_members']
                    it_mni = np.zeros(n_snp)
                    for i_snp in range(n_snp):
                        it_mni[i_snp] = igraph.compare_communities(it_solution[i_snp], ground_truth[i_snp], method='nmi')
                    mni_matrix[i, :] = np.array(it_mni)
                except Exception as e:
                    raise RuntimeError("Dataset {0}, snp {1}, it {2}, raise exception {3}".format(self.dataset['_id'], i_snp, it['_id'], e))

            return mni_matrix
        else:
            raise RuntimeError("{0} has no ground truth".format(self.dataset['_id']))

    def get_best_mni_matrix(self) -> np.array:
        """
        (nº_iterations, nº_snpshot)
        :return:
        """
        if 'memebers' in self.dataset:
            ground_truth = self.dataset['memebers']
        else:
            ground_truth = self.dataset['members']

        n_snapshots = self.dataset['snapshot_count']
        number_iterations = len(self.iteration_list)

        best_mni_matrix = np.zeros((number_iterations, n_snapshots))
        for n_it in range(number_iterations):
            paretos = self._extract_pareto(n_it)

            for n_snp, pareto_snp in enumerate(paretos):
                max_mni = 0
                for ind in pareto_snp:
                    members = decode(ind)
                    actual_mni = igraph.compare_communities(members, ground_truth[n_snp], method='nmi')
                    max_mni = max(max_mni, actual_mni)

                best_mni_matrix[n_it, n_snp] = max_mni

        return best_mni_matrix

    def _make_hypervolume_matrix(self, max_generations: int) -> np.array:
        n_snapshots = self.dataset['snapshot_count']
        number_iterations = len(self.iteration_list)

        hv_matrix = np.zeros((number_iterations, n_snapshots, max_generations))
        for n_it, iteration in enumerate(self.iteration_list):
            snapshots = iteration['snapshots']
            for n_sn, hypervolumes in enumerate(snapshots):
                hv_np = np.array(hypervolumes)
                actual_generations = hv_np.shape[0]
                hv_np = np.pad(hv_np, (0, max_generations - actual_generations), 'maximum')

                hv_matrix[n_it, n_sn, :] = hv_np

        return hv_matrix

    def _make_pop_matrix(self, max_generations: int, itype: str) -> np.array:
        n_snapshots = self.dataset['snapshot_count']
        number_iterations = len(self.iteration_list)

        pop_matrix = np.zeros((number_iterations, n_snapshots, max_generations))
        for n_it, iteration in enumerate(self.iteration_list):
            snp_types = iteration["execution_info"]["population_types"]
            for n_snp, pop_all_types in enumerate(snp_types):
                pop_type = np.array(pop_all_types[itype])
                actual_generations = pop_type.shape[0]
                pop_type = np.pad(pop_type, (0, max_generations - actual_generations), 'edge')

                pop_matrix[n_it, n_snp, :] = pop_type

        return pop_matrix

    def _make_whole_pop_matrix(self, max_generations_snp: np.ndarray, itype: str) -> np.array:
        """

        """
        max_generations = int(np.sum(max_generations_snp))
        number_iterations = len(self.iteration_list)

        pop_matrix = np.zeros((number_iterations, max_generations))
        for n_it, iteration in enumerate(self.iteration_list):

            snp_types = iteration["execution_info"]["population_types"]
            it_pop_list = np.zeros(max_generations)
            s_index = 0

            for n_snp, pop_all_types in enumerate(snp_types):
                pop_type = np.array(pop_all_types[itype])
                e_index = s_index + max_generations_snp[n_snp]

                actual_generations = pop_type.shape[0]
                values_to_add = max_generations_snp[n_snp] - actual_generations

                pop_type = np.pad(pop_type, (0, values_to_add), 'edge')
                it_pop_list[s_index:e_index] = pop_type
                s_index += max_generations_snp[n_snp]

            pop_matrix[n_it, :] = it_pop_list

        return pop_matrix

    def _extract_pareto_hv(self, n_iter: int) -> List[np.array]:
        paretos = self._extract_pareto(n_iter)

        paretos_fitness = []
        for actual_pareto in paretos:
            n_points = len(actual_pareto)
            fitness_1 = [0.0] * n_points
            fitness_2 = [0.0] * n_points
            for n_ind, ind in enumerate(actual_pareto):
                fitness_1[n_ind] = ind.fitness.values[0]
                fitness_2[n_ind] = ind.fitness.values[1]

            fitness_all = np.zeros((n_points, 2))
            fitness_all[:, 0] = np.array(fitness_1)
            fitness_all[:, 1] = np.array(fitness_2)
            paretos_fitness.append(fitness_all)

        return paretos_fitness

    def _extract_pareto_modularity(self, n_iter: int) -> List[np.array]:
        paretos = self._extract_pareto(n_iter)
        dataset_obj = db_queries.get_dataset_obj(self.dataset['_id'])

        paretos_fitness = []
        for n_snp, actual_pareto in enumerate(paretos):
            fitness_all = []
            for ind in actual_pareto:
                fitness_all.append(modularity_individual(ind, dataset_obj.snapshots[n_snp]))

            paretos_fitness.append(np.array(fitness_all))

        return paretos_fitness

    def _extract_pareto(self, n_iter: int) -> List[np.array]:
        if "version" not in self.iteration_list[n_iter]:
            paretos_individuals = self.iteration_list[n_iter]['paretos']
            paretos = []
            for actual_pareto_compress in paretos_individuals:
                actual_pareto = [pickle.loads(ind) for ind in actual_pareto_compress]
                paretos.append(actual_pareto)

            return paretos
        elif self.iteration_list[n_iter]['version'] == "v2":
            return db_queries.get_pareto(self.iteration_list[n_iter]['_id'])
        else:
            raise RuntimeError("Unknown version of iteration {0}".format(self.iteration_list[n_iter]['version']))
