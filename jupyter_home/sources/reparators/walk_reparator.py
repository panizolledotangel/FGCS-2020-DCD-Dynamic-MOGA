import random
from typing import Dict, List

import igraph

from sources.gas import creator
from sources.gas.auxiliary_funtions import random_neighbor
from sources.reparators.reparator_interface import ReparatorInterface


class RandomWalkReparator(ReparatorInterface):

    @classmethod
    def load_from_dict(cls, d: Dict):
        return RandomWalkReparator(d['name'], d['num_walkers'], d['num_steps'])

    def __init__(self, name: str,  num_walkers: int, num_steps: int):
        super().__init__(name)
        self.num_walkers = num_walkers
        self.num_steps = num_steps

        self.walkers_done = None

    def make_dict(self) -> Dict:
        d = super().make_dict()
        d.update({
            "num_walkers": self.num_walkers,
            "num_steps": self.num_steps
        })
        return d

    def set_snapshots(self, actual_snapshot: igraph.Graph, previous_snapshot: igraph.Graph,
                      previous_solution: List[int]):

        super().set_snapshots(actual_snapshot, previous_snapshot, previous_solution)
        self.walkers_done = {}

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
                        selected_index = self._random_neighbor_walks(index, neighbors)
                        new_individual[index] = selected_index
                        n_gen_repair += 1
                else:
                    # old node connected to have disappeared between timestamps
                    selected_index = self._random_neighbor_walks(index, neighbors)
                    new_individual[index] = selected_index
                    n_gen_repair += 1
            else:
                # the node is new in this timestamp
                selected_index = self._random_neighbor_walks(index, neighbors)
                new_individual[index] = selected_index
                n_gen_repair += 1

        return new_individual, n_gen_repair

    def _random_neighbor_walks(self, start_index: int, neighbors: List[int]) -> int:
        if len(neighbors) > 1:
            neighbors_selected = None
            best_percent = 0
            best_comm = None

            communitiess_walks = self._get_community_walks(start_index)
            for n in neighbors:
                n_comm_id = self.members_new_old[n]
                if n_comm_id in communitiess_walks and communitiess_walks[n_comm_id] > best_percent:
                    best_comm = n_comm_id
                    neighbors_selected = []

                if best_comm == n_comm_id:
                    neighbors_selected.append(n)

            return neighbors_selected[random.randint(0, len(neighbors_selected) - 1)]
        else:
            return neighbors[0]

    def _get_community_walks(self,  start_index: int) -> Dict[int, float]:
        if start_index not in self.walkers_done:
            total = {}
            for i in range(0, self.num_walkers):
                actual = self._do_random_walk(start_index)

                for com_id in actual.keys():
                    if com_id not in total:
                        total[com_id] = actual[com_id]/self.num_walkers
                    else:
                        total[com_id] += actual[com_id]/self.num_walkers
            self.walkers_done[start_index] = total

        return self.walkers_done[start_index]

    def _do_random_walk(self, start_index: int) -> Dict[int, int]:
        steps_done = 0
        communities = {}

        actual_index = start_index
        while steps_done <= self.num_steps:
            actual_index = random_neighbor(self.actual_snapshot, actual_index)
            com_id = self.members_new_old[actual_index]

            if com_id not in communities:
                communities[com_id] = 0

            communities[com_id] += 1
            steps_done += 1
        return communities
