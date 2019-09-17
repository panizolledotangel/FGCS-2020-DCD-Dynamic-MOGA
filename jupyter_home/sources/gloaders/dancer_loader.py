import re
from logging import warning
from typing import List, Dict, Tuple, Set
from os import listdir
from os.path import isfile, join, exists

import igraph

from sources.gloaders.loader_interface import LoaderInterface


class DancerLoader(LoaderInterface):

    VERTEX_STATE = 0
    EDGES_STATE = 1
    ERROR = 2

    @classmethod
    def load_from_dict(cls, d: Dict):
        kwargs = {
            'dataset_directory': d["dataset_directory"]
        }
        return DancerLoader(**kwargs)

    @classmethod
    def _process_vertex(cls, line: str, snapshot: igraph.Graph, members: List[int], last_id: int,
                        name_id_table: Dict[str, int], communities_ids: Set[int]):

        first_index = line.find(";")
        last_index = line.rfind(";")

        if first_index != -1 and last_index != -1:
            node_name = line[0:first_index]
            community_id = int(line[last_index+1:-1])

            snapshot.add_vertex(name=node_name)
            name_id_table[node_name] = last_id
            members.append(community_id)
            communities_ids.add(community_id)

            last_id += 1
            return last_id
        else:
            raise RuntimeError("format error on line '{0}'".format(line))

    @classmethod
    def _process_edge(cls, line: str, name_id_table: Dict[str, int], tuplelist: List[Tuple[int, int]]):
        first_index = line.find(";")

        if first_index != -1:
            source_name = line[0:first_index]
            target_name = line[first_index + 1:-1]

            tuplelist.append((name_id_table[source_name], name_id_table[target_name]))
        else:
            raise RuntimeError("format error")

    @classmethod
    def _remove_isolated_nodes(cls, g: igraph.Graph) -> igraph.Graph:
        components = g.components(mode=igraph.WEAK)

        max_size = 0
        nodes = None
        for actual_component in components:
            size = len(actual_component)
            if size > max_size:
                max_size = size
                nodes = actual_component

        return g.subgraph(nodes)

    @classmethod
    def _process_graph_file(cls, file_path: str) -> Tuple[igraph.Graph, int, int, List[int], int]:

        snapshot = igraph.Graph(directed=False)

        members = []
        comms_ids = set()
        last_id = 0
        name_id_table = {}

        tuplelist = []

        actual_state = -1

        with open(file_path, 'r') as f:
            for line in f:

                if line.startswith("# Vertices"):
                    actual_state = cls.VERTEX_STATE
                elif line.startswith("# Edges"):
                    actual_state = cls.EDGES_STATE

                elif not line.startswith("#"):

                    if actual_state == cls.VERTEX_STATE:
                        try:
                            last_id = cls._process_vertex(line, snapshot, members, last_id, name_id_table, comms_ids)
                        except Exception as e:
                            raise RuntimeError("Exception: {0}, reach at line '{1}', on file {2}'"
                                               .format(e, line, file_path))

                    elif actual_state == cls.EDGES_STATE:
                        try:
                            cls._process_edge(line, name_id_table, tuplelist)
                        except Exception as e:
                            raise RuntimeError("Exception: {0}, reach at line '{1}', on file {2}'"
                                               .format(e, line, file_path))
                    else:
                        raise RuntimeError("Impossible state: {0}, reach at line '{1}', on file {2}'"
                                           .format(actual_state, line, file_path))

        snapshot.add_edges(tuplelist)
        snapshot.simplify(multiple=True)
        snapshot = cls._remove_isolated_nodes(snapshot)

        if snapshot.vcount() != len(members):
            warning("dataset with more than one component, beware of ground truth!")
            members = cls._update_members(snapshot, members, name_id_table)

        n_comms = len(comms_ids)
        return snapshot, snapshot.vcount(), snapshot.ecount(), members, n_comms

    @classmethod
    def _read_n_snapshots(cls, dir_path: str) -> int:
        whole_path = join(dir_path, "parameters")

        if exists(whole_path):
            with open(whole_path, 'r') as f:
                for line in f:
                    if line.startswith("nbTimestamps"):
                        index = line.find(":")
                        return int(line[index + 1:-1])

            raise RuntimeError("'nbTimestamps' not found on file '{0}'".format(whole_path))
        else:
            raise FileNotFoundError("parameters file not found in directory '{0}'".format(dir_path))

    @classmethod
    def _process_directory(cls, dir_path: str) -> Tuple[List[igraph.Graph], List[int], List[int], List[List[int]], List[int]]:
        n_snapshots = cls._read_n_snapshots(dir_path)

        snapshots = [igraph.Graph()]*n_snapshots
        n_nodes = [0]*n_snapshots
        n_edges = [0]*n_snapshots

        comms = [[]]*n_snapshots
        n_comms = [0]*n_snapshots

        re_snp_file = re.compile("(t)(\d*)(.graph)")
        for f in listdir(dir_path):
            match = re_snp_file.match(f)
            whole_path = join(dir_path, f)

            if isfile(whole_path) and match is not None:
                g, nodes, edges, members, snp_comms = cls._process_graph_file(whole_path)

                n_snp = int(match.group(2))
                snapshots[n_snp] = g
                n_nodes[n_snp] = nodes
                n_edges[n_snp] = edges
                comms[n_snp] = members
                n_comms[n_snp] = snp_comms

        return snapshots, n_nodes, n_edges, comms, n_comms

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_datatset(self, **kwargs):
        snapshots, n_nodes, n_edges, communities, n_comms = DancerLoader._process_directory(kwargs["dataset_directory"])

        n_ts = len(snapshots)

        info = {
            "dataset_directory": kwargs["dataset_directory"],
            "snapshot_count": n_ts,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "ground_truth": True,
            "members": communities,
            "n_communites": n_comms
        }
        return snapshots, n_ts, n_nodes, n_edges, info, communities, n_comms

    @classmethod
    def _update_members(cls, new_snapshot: igraph.Graph, old_members: List[int], name_id_table: Dict[str, int]) -> List[int]:
        new_members = [-1] * new_snapshot.vcount()

        for v in new_snapshot.vs:
            node_old_id = name_id_table[v['name']]
            new_members[v.index] = old_members[node_old_id]

        return new_members

