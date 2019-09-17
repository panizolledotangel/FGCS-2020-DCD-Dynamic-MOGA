import random
from typing import Dict, List

import sources.gas.auxiliary_funtions as auxF
from sources.gas import creator
from sources.reparators.reparator_interface import ReparatorInterface


class GreedyReparator(ReparatorInterface):

    @classmethod
    def load_from_dict(cls, d: Dict):
        return GreedyReparator(d['name'], d['num_iter'])

    def __init__(self, name: str, num_iter=0):
        super().__init__(name)
        self.num_iter = num_iter

    def make_dict(self) -> Dict:
        d = super().make_dict()
        d['num_iter'] = self.num_iter
        return d

    def repair(self, individual):
        n_gen_repair = 0

        actual_size = self.actual_snapshot.vcount()
        new_individual = creator.Individual([None] * actual_size)

        for index in range(actual_size):
            neighbors = self.actual_snapshot.neighborhood(vertices=index)

            if index in self.equivalence_new_old:
                # the node existed in the previous timestamp
                old_index = self.equivalence_new_old[index]
                old_connected_index = individual[old_index]

                if old_connected_index in self.equivalence_old_new:
                    # both nodes have equivalence check if there are still connected
                    new_connected_index = self.equivalence_old_new[old_connected_index]

                    if new_connected_index in neighbors:
                        new_individual[index] = new_connected_index
                    else:
                        selected_index = self._random_neighbor_most(neighbors)
                        new_individual[index] = selected_index
                        n_gen_repair += 1
                else:
                    # old node connected to have disappeared between timestamps
                    selected_index = self._random_neighbor_most(neighbors)
                    new_individual[index] = selected_index
                    n_gen_repair += 1
            else:
                # the node is new in this timestamp
                selected_index = self._random_neighbor_most(neighbors)
                new_individual[index] = selected_index
                n_gen_repair += 1

        auxF.do_convergence_steps(new_individual, self.num_iter, self.actual_snapshot)
        return new_individual, n_gen_repair

    def _random_neighbor_most(self, neighbors_index: List[int]) -> int:
        if len(neighbors_index) > 1:
            communities_aux = {}
            best_n_users = 0
            best_comm_id = None

            for n in neighbors_index:
                n_comm_id = self.members_new_old[n]

                if n_comm_id not in communities_aux:
                    communities_aux[n_comm_id] = {
                        "value": 0,
                        "members": []
                    }

                communities_aux[n_comm_id]['value'] += 1
                communities_aux[n_comm_id]['members'].append(n)

                aux_value = communities_aux[n_comm_id]['value']
                if aux_value > best_n_users:
                    best_n_users = communities_aux[n_comm_id]['value']
                    best_comm_id = n_comm_id

            neighbors_selected = communities_aux[best_comm_id]['members']
            return neighbors_selected[random.randint(0, len(neighbors_selected) - 1)]
        else:
            return neighbors_index[0]
