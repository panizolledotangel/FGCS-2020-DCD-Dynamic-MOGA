from abc import ABCMeta, abstractmethod
from typing import Dict, List

import igraph

from sources.gas.auxiliary_funtions import find_equivalences_and_members


class ReparatorInterface:
    _metaclass__ = ABCMeta

    def __init__(self, name: str):
        self.name = name

        self.actual_snapshot = None
        self.previous_snapshot = None
        self.previous_solution = None

        self.equivalence_old_new = None
        self.equivalence_new_old = None
        self.members_new_old = None

    def set_snapshots(self, actual_snapshot: igraph.Graph, previous_snapshot: igraph.Graph,
                      previous_solution: List[int]):

        self.actual_snapshot = actual_snapshot
        self.previous_snapshot = previous_snapshot
        self.previous_solution = previous_solution

        eq_old_new, eq_new_old, members = find_equivalences_and_members(self.actual_snapshot, self.previous_snapshot,
                                                                        self.previous_solution)
        self.equivalence_old_new = eq_old_new
        self.equivalence_new_old = eq_new_old
        self.members_new_old = members

    def get_name(self):
        return self.name

    def make_dict(self) -> Dict:
        d = {
            "name": self.name
        }
        return d

    @abstractmethod
    def repair(self, individual): raise NotImplementedError
