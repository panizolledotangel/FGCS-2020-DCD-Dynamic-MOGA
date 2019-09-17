import warnings
from typing import List, Dict, Tuple
from os import listdir
from os.path import join

import igraph

from sources.gloaders.loader_interface import LoaderInterface


class TrevianLoader(LoaderInterface):

    @classmethod
    def load_from_dict(cls, d: Dict):
        kwargs = {
            'graphml_directory': d["graphml_directory"],
            'communities_directory': d["communities_directory"]
        }
        return TrevianLoader(**kwargs)

    @classmethod
    def _process_directories(cls, graphml_dir: str, communities_dir: str) -> Tuple[List[igraph.Graph], List[int], List[int], List[List[int]], List[int]]:
        communities_dicts = cls._process_communities_dir(communities_dir)
        snapshots, n_nodes, n_edges, communities, n_comms = cls._process_graphml_dir(graphml_dir, communities_dicts)

        return snapshots, n_nodes, n_edges, communities, n_comms

    @classmethod
    def _process_graphml_dir(cls, graphml_dir: str, communities_dict: List[Dict[str, int]]) -> Tuple[List[igraph.Graph], List[int], List[int], List[List[int]], List[int]]:
        files = list(listdir(graphml_dir))

        n_ts = len(files)
        snapshots = [None]*n_ts
        members_list = [None]*n_ts
        n_comms_list = [None]*n_ts

        for f_name in files:
            try:
                # load on the correct position at the array
                point_index = f_name.rfind(".")
                dash_index = f_name.rfind("-")
                n_index = int(f_name[dash_index + 1:point_index]) - 1

                # get the communities dict
                actual_comms_dict = communities_dict[n_index]

                # load graph, get biggest component, remove the nodes without guild
                whole_graph = igraph.Graph.Read_GraphML(join(graphml_dir, f_name))
                whole_graph.to_undirected()

                nodes_to_remove = []
                for n in whole_graph.vs:
                    if not n["label"] in actual_comms_dict.keys():
                        nodes_to_remove.append(n.index)
                    else:
                        n["name"] = n["label"]
                whole_graph.delete_vertices(nodes_to_remove)

                components = whole_graph.clusters()
                biggest_c = components.giant()
                biggest_c.simplify(multiple=True)

                # Make the members vector
                actual_members, actual_ncomms = cls._make_members_vector(biggest_c, actual_comms_dict)

                # Store the results
                snapshots[n_index] = biggest_c
                members_list[n_index] = actual_members
                n_comms_list[n_index] = actual_ncomms
            except KeyError as e:
                raise RuntimeError("Missing community for node {0} in file {1}".format(e, f_name))
            except Exception as e:
                raise RuntimeError("Exception {0} arise while parsing file {1}".format(e, f_name))

        return snapshots, [g.vcount() for g in snapshots], [g.ecount() for g in snapshots], members_list, n_comms_list

    @classmethod
    def _make_members_vector(cls, snapshot: igraph.Graph, comms_dict: Dict[str, int]) -> Tuple[List[int], int]:
        members = [-1]*snapshot.vcount()
        for n in snapshot.vs:
            members[n.index] = comms_dict[n["label"]]

        return members, len(set(members))

    @classmethod
    def _process_communities_dir(cls, comms_dir: str) -> List[Dict[str, int]]:
        files = list(listdir(comms_dir))
        communities_dicts = [None]*len(files)

        for f_name in files:
            # load on the correct position at the array
            point_index = f_name.rfind(".")
            dash_index = f_name.rfind("-")
            n_index = int(f_name[dash_index + 1:point_index]) - 1

            # read all the communities
            snp_dict = cls._process_communities_file(join(comms_dir, f_name))
            communities_dicts[n_index] = snp_dict

        return communities_dicts

    @classmethod
    def _process_communities_file(cls, comms_file_path: str) -> Dict[str, int]:
        coms_dict = {}
        n_line = 0

        with open(comms_file_path, 'r') as f:
            for line in f:
                members = line.split()
                for user in members:
                    coms_dict[user] = n_line

                n_line += 1

        return coms_dict

    def __init__(self, **kwargs):
        """

        :param kwargs: must receive  'graphml_directory' and 'communities_directory' with the path of the directory
        containing the graphml files and the communities files
        """
        super().__init__(**kwargs)

    def load_datatset(self, **kwargs):
        snapshots, n_nodes, n_edges, communities, n_comms = TrevianLoader._process_directories(kwargs["graphml_directory"], kwargs["communities_directory"])

        n_ts = len(snapshots)

        info = {
            "graphml_directory": kwargs["graphml_directory"],
            "communities_directory": kwargs["communities_directory"],
            "snapshot_count": n_ts,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "ground_truth": True,
            "members": communities,
            "n_communites": n_comms
        }
        return snapshots, n_ts, n_nodes, n_edges, info, communities, n_comms
