import warnings
from os. path import join
from abc import ABCMeta, abstractmethod
from random import randint
from typing import List

import igraph
import numpy as np
import seaborn as sns; sns.set()

import sources.gloaders.distribution_colors as dcolors
import sources.mongo_connection.mongo_queries as db_queries
from sources.gas.auxiliary_funtions import average_odf_and_internal_density, average_odf_and_community_score


class LoaderInterface:
    _metaclass__ = ABCMeta

    @classmethod
    def _calculate_top_names(cls, snapshot: igraph.Graph, n_tops: int) -> List[str]:
        centralities = np.array(snapshot.eigenvector_centrality())
        order_centrality_index = np.argsort(centralities)[::-1]

        top_names = [snapshot.vs[index]["name"] for index in order_centrality_index[0:n_tops]]
        return top_names

    @classmethod
    def _number_missing_edges(cls, actual_edges: igraph.EdgeSeq, actual_nodes: igraph.VertexSeq,
                              previous_edges: igraph.EdgeSeq, previous_nodes: igraph.VertexSeq) -> int:
        actual_new_edges = 0
        for e in actual_edges:
            source = e.source
            target = e.target

            source_name = actual_nodes[source]["name"]
            target_name = actual_nodes[target]["name"]

            source_n_prev_ts = previous_nodes.select(name=source_name)
            target_n_prev_ts = previous_nodes.select(name=target_name)

            if len(source_n_prev_ts) > 0 and len(target_n_prev_ts) > 0:
                source_id_prev_ts = source_n_prev_ts[0].index
                target_id_prev_ts = target_n_prev_ts[0].index

                prev_edge = previous_edges.select(_between=((source_id_prev_ts,), (target_id_prev_ts,)))
                if len(prev_edge) == 0:
                    actual_new_edges += 1
            else:
                actual_new_edges += 1
        return actual_new_edges

    def __init__(self, **kwargs):
        snapshots, n_ts, n_nodes, n_edges, info, communities, n_comms = self.load_datatset(**kwargs)

        self.snapshots = snapshots
        self.n_ts = n_ts
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        self.communities = communities
        self.n_comms = n_comms
        self.info = info

        if communities:
            # self.available_colors = ['%06X' % randint(0, 0xFFFFFF) for _ in range(max(self.n_comms) + 1)]
            self.available_colors = ['green', 'blue', 'red', 'yellow', 'purple']

    def summary(self):
        c_info = dict(self.info)
        del(c_info["members"])
        print(c_info)

    def compute_number_connected_components(self):
        components = [len(g.components(mode=igraph.WEAK)) for g in self.snapshots]
        return components

    def store_or_update_db(self, dataset_name: str):
        if db_queries.get_dataset(dataset_name) is None:
            if db_queries.count_dataset(self) == 0:
                db_queries.save_dataset(dataset_name, self)
            else:
                ids = [c['_id'] for c in cursor]
                warnings.warn("Petition ignored already exists a dataset document in the db with same parameters, "
                              "dataset name(s) is {0}".format(ids))
        else:
            db_queries.update_datatse(dataset_name, self)

    def get_snapshots(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else self.n_ts
        return self.snapshots[n_ts_min:end]

    def get_communities(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else self.n_ts
        return self.communities[n_ts_min:end]

    def get_dataset_info(self):
        return self.info

    def remove_snapshots(self, snaphsot_ids: List[int]):
        for i in sorted(snaphsot_ids, reverse=True):
            del self.snapshots[i]
            del self.n_nodes[i]
            del self.n_edges[i]

        self.n_ts -= len(snaphsot_ids)
        self.info['snapshot_count'] = self.n_ts
        self.info['n_nodes'] = self.n_nodes
        self.info['n_edges'] = self.n_edges

    def compute_graph_edit_distance(self) -> np.array:
        edit_distance = np.zeros(self.n_ts)

        for s_i in range(1, self.n_ts):
            # Nodes changes
            n1 = set(self.snapshots[s_i].vs["name"])
            n2 = set(self.snapshots[s_i - 1].vs["name"])

            n_v_insertions = len(n1.difference(n2))
            n_v_deletions = len(n2.difference(n1))

            # Edges changes
            actual_edges = self.snapshots[s_i].es
            actual_nodes = self.snapshots[s_i].vs

            previous_edges = self.snapshots[s_i - 1].es
            previous_nodes = self.snapshots[s_i - 1].vs

            n_e_insertions = self._number_missing_edges(actual_edges, actual_nodes, previous_edges, previous_nodes)
            n_e_deletions = self._number_missing_edges(previous_edges, previous_nodes, actual_edges, actual_nodes)

            edit_distance[s_i] = (n_v_insertions + n_v_deletions + n_e_insertions + n_e_deletions)

        return edit_distance

    def get_modularity_values(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        modularities = [0.0] * n_ts
        if self.communities:
            for i in range(n_ts):
                modularities[i] = snapshots[i].modularity(self.communities[i])
        else:
            for i in range(n_ts):
                comms = snapshots[i].community_infomap()
                modularities[i] = snapshots[i].modularity(comms)

        return modularities

    def compute_objectives_solution(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        objectives = [0.0] * n_ts
        if self.communities:
            for i in range(n_ts):
                objectives[i] = average_odf_and_internal_density(snapshots[i], self.communities[i])
        else:
            for i in range(n_ts):
                comms = snapshots[i].community_infomap()
                objectives[i] = average_odf_and_internal_density(snapshots[i], comms)

        return objectives

    def compute_objectives_solution_2(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        objectives = [0.0] * n_ts
        if self.communities:
            for i in range(n_ts):
                objectives[i] = average_odf_and_community_score(snapshots[i], self.communities[i])
        else:
            for i in range(n_ts):
                comms = snapshots[i].community_infomap()
                objectives[i] = average_odf_and_community_score(snapshots[i], comms)

        return objectives

    @abstractmethod
    def load_datatset(self, **kwargs) -> (list, int, list, list): raise NotImplementedError

    def save_graphs_img(self, path: str, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else self.n_ts
        snapshots = self.snapshots[n_ts_min:end]
        coms = self.communities[n_ts_min:end]

        for i in range(len(snapshots)):
            img_path = join(path, "snapshot_{0}.png".format(i))

            for v in snapshots[i].vs:
                if self.communities:
                    node_id = v.index
                    community_id = int(coms[i][node_id])
                    v['color'] = self.available_colors[community_id]
                else:
                    v['color'] = 'green'

            visual_style = {"bbox": (800, 600), "margin": 60, "vertex_size": 8, "vertex_label_size": 11,
                            "vertex_color": snapshots[i].vs['color'], "edge_width": 0.5,
                            "layout": snapshots[i].layout_auto(), "vertex_label": snapshots[i].vs['name']}

            igraph.plot(snapshots[i], img_path, **visual_style)

    def plot_neighbors_change_ts(self, axs, axs_legend, n_ts_min=0, n_ts_max=-1):
        width = 100
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 16}

        end = n_ts_max if n_ts_max > 0 else self.n_ts
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        med = []
        xs = []
        scales = []
        for s_i in range(1, n_ts):
            nodes_changes = []
            for i in range(snapshots[s_i].vcount()):
                n1 = snapshots[s_i].vs[i]
                n2 = snapshots[s_i - 1].vs.select(name=n1["name"])
                if len(n2) > 0:
                    n2 = n2[0]
                    s1 = set([x["name"] for x in n1.neighbors()])
                    s2 = set([x["name"] for x in n2.neighbors()])

                    neighbors_missing = max(len(s1.difference(s2)), len(s2.difference(s1)))
                    missing_value = (neighbors_missing / max(len(s2), len(s1)))*100
                    missing_value = round(missing_value)
                    nodes_changes.append(missing_value)

            values, classes = np.histogram(nodes_changes, bins=10)
            values = values/len(nodes_changes)

            n_classes = []
            n_values = []
            for i in range(len(values)):
                if values[i] > 0.0:
                    n_classes.append(classes[i])
                    n_values.append(values[i])

            med.extend(n_classes)
            xs.extend([s_i]*len(n_classes))
            scales.extend(n_values)

        colors = np.zeros(len(scales)*3).reshape(len(scales), 3)
        for i in range(len(scales)):
            colors[i,:] = dcolors.select_color_10(scales[i])

        axs.scatter(xs, med, c=colors, edgecolors='#252525', marker='s', s=width)

        axs.set_xlabel("id timestamp number", fontdict=font)
        axs.set_ylabel("% of missing neighbors", fontdict=font)

        dcolors.print_legend_10(axs_legend)

    def plot_neighbors_change_heatmap(self, n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else self.n_ts
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        # find node with the highest id for the picture size
        max_n_nodes = -1
        for g in snapshots:
            g_max_name = -1
            for v in g.vs:
                v_name = int(v["name"])
                if v_name > g_max_name:
                    g_max_name = v_name

            if g_max_name > max_n_nodes:
                max_n_nodes = g_max_name

        img = np.zeros((max_n_nodes + 1, n_ts)) - 1

        for s_i in range(1, n_ts):
            for i in range(snapshots[s_i].vcount()):
                n1 = snapshots[s_i].vs[i]
                n2 = snapshots[s_i - 1].vs.select(name=n1["name"])

                missing_value = 0
                if len(n2) > 0:
                    n2 = n2[0]
                    s1 = set([x["name"] for x in n1.neighbors()])
                    s2 = set([x["name"] for x in n2.neighbors()])

                    neighbors_missing = max(len(s1.difference(s2)), len(s2.difference(s1)))
                    missing_value = (neighbors_missing / max(len(s2), len(s1)))

                img[int(n1["name"]), s_i] = missing_value

        sns.heatmap(img, vmin=-0.5, vmax=1, center=0)

    def plot_node_degree_ts(self, axs, axs_legend, n_ts_min=0, n_ts_max=-1):
        width = 100
        bins = [1, 2, 4, 6, 8, 15, 25, 100, 200, 500, 1000]

        end = n_ts_max if n_ts_max > 0 else self.n_ts
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        med = []
        xs = []
        scales = []
        for s_i in range(n_ts):
            nodes_degrees = snapshots[s_i].degree(snapshots[s_i].vs, mode='ALL')

            values, classes = np.histogram(nodes_degrees, bins=bins)
            values = values/len(nodes_degrees)

            n_classes = []
            n_values = []
            for i in range(len(values)):
                if values[i] > 0.0:
                    n_classes.append(classes[i])
                    n_values.append(values[i])

            med.extend(n_classes)
            xs.extend([s_i]*len(n_classes))
            scales.extend(n_values)

        colors = np.zeros(len(scales)*3).reshape(len(scales), 3)
        for i in range(len(scales)):
            colors[i,:] = dcolors.select_color_10(scales[i])

        axs.scatter(xs, med, c=colors, edgecolors='#252525', marker='s', s=width)
        axs.set_yscale('log')
        axs.set_xlabel("id timestamp")
        axs.set_ylabel("node degree")

        dcolors.print_legend_10(axs_legend)

    def plot_missing_nodes(self, axs, n_ts_min=0, n_ts_max=-1):
        width = 0.8
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        med = np.zeros(n_ts - 1)
        for s_i in range(1, n_ts):
            n1 = set(snapshots[s_i].vs["name"])
            n2 = set(snapshots[s_i-1].vs["name"])

            med[s_i-1] = len(n1.difference(n2))/len(n1) * 100

        axs.bar(range(1, n_ts), med, width)
        axs.axhline(np.median(med), color="orange")

        for i in range(n_ts-1):
            axs.text(i + width, med[i], "{0: .2f}".format(med[i]))

        axs.set_xlabel("id timestamp")
        axs.set_ylabel("% node change")

    def plot_number_nodes_ts(self, axs,  n_ts_min=0, n_ts_max=-1):
        width = 0.8
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        new_nodes = np.zeros(n_ts)
        old_nodes = np.zeros(n_ts)
        for s_i in range(1, n_ts):
            n1 = set(snapshots[s_i].vs["name"])
            n2 = set(snapshots[s_i - 1].vs["name"])

            ts_new = n1.difference(n2)
            ts_old = n2.difference(n1)

            new_nodes[s_i] = len(ts_new)
            old_nodes[s_i] = len(ts_old)

        sizes = np.array(self.n_nodes[n_ts_min:end])
        continuous_nodes = sizes - new_nodes

        p0 = axs.bar(range(n_ts), -old_nodes, width, color='tomato')
        p1 = axs.bar(range(n_ts), continuous_nodes, width, color='royalblue')
        p2 = axs.bar(range(n_ts), new_nodes, width, bottom=continuous_nodes, color='seagreen')

        axs.legend((p0[0], p1[0], p2[0]), ('Lost', 'Stay', 'New'))

        axs.set_xlabel("id timestamp")
        axs.set_ylabel("number of new/old nodes")

    def plot_number_edges_ts(self, axs,  n_ts_min=0, n_ts_max=-1):
        width = 0.8
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        new_edges = np.zeros(n_ts)
        lost_edges = np.zeros(n_ts)
        for s_i in range(1, n_ts):
            actual_edges = snapshots[s_i].es
            actual_nodes = snapshots[s_i].vs

            previous_edges = snapshots[s_i - 1].es
            previous_nodes = snapshots[s_i - 1].vs

            new_edges[s_i] = LoaderInterface._number_missing_edges(actual_edges, actual_nodes,
                                                                   previous_edges, previous_nodes)

            lost_edges[s_i] = LoaderInterface._number_missing_edges(previous_edges, previous_nodes,
                                                                    actual_edges, actual_nodes)

        sizes = np.array(self.n_edges[n_ts_min:end])
        continuous_edges = sizes - new_edges

        p0 = axs.bar(range(n_ts), -lost_edges, width, color='tomato')
        p1 = axs.bar(range(n_ts), continuous_edges, width, color='royalblue')
        p2 = axs.bar(range(n_ts), new_edges, width, bottom=continuous_edges, color='seagreen')

        axs.legend((p0[0], p1[0], p2[0]), ('Lost', 'Stay', 'New'))

        axs.set_xlabel("id timestamp")
        axs.set_ylabel("number of new/old edges")

    def plot_modularity_infomap(self, axs,  n_ts_min=0, n_ts_max=-1):
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        modularities = self.get_modularity_values(n_ts_min, n_ts_max)

        axs.plot(range(n_ts), modularities)
        axs.set_xlabel("id timestamp")
        axs.set_ylabel("modularity value")

    def plot_important_node_variation(self, axs, n_percent: float,  n_ts_min=0, n_ts_max=-1):
        variations = self.important_nodes_variation(n_percent, n_ts_min, n_ts_max)

        variation_ts_percent = []
        for ts_variation in variations:
            stay_count = 0
            for i in range(len(ts_variation)):
                if ts_variation[i]:
                    stay_count += 1
            variation_ts_percent.append(stay_count/len(ts_variation))

        axs.plot(range(1, len(variations)+1), variation_ts_percent)
        axs.set_xlabel("id timestamp")
        axs.set_ylabel("% of important nodes mantained")

    def important_nodes_variation(self, n_percent: float, n_ts_min=0, n_ts_max=-1) -> List[List[bool]]:
        end = n_ts_max if n_ts_max > 0 else (self.n_ts + 1)
        snapshots = self.snapshots[n_ts_min:end]
        n_ts = len(snapshots)

        variations = []

        n_nodes = round(self.info["n_nodes"][0] * n_percent)
        last_top = LoaderInterface._calculate_top_names(snapshots[0], n_nodes)
        for i in range(1, n_ts):
            actual_variation = []
            for n_name in last_top:
                n_found = snapshots[i].vs.select(name=n_name)
                actual_variation.append(len(n_found) > 0)
            variations.append(actual_variation)

            n_nodes = round(self.info["n_nodes"][i] * n_percent)
            last_top = LoaderInterface._calculate_top_names(snapshots[i], n_nodes)
        return variations

