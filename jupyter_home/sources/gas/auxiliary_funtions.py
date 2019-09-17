import math
import random
from itertools import chain
from operator import attrgetter
from typing import List, Dict, Tuple
import sys

import numpy as np
import igraph
from deap.tools.emo import sortNondominated, assignCrowdingDist, selNSGA2

from sources.gas import creator, RANDOM, RANDOM_RANDOM, IMMIGRANT, IMMIGRANT_IMMIGRANT, IMMIGRANT_RANDOM


# Operators for individuals

def decode(chrom):
    """
    Returns the communities of a locus-based adjacency codification
    in a vector of int where each position is a node id and the value
    of that position the id of the community where it belongs. To position
    with the same number means that those two nodes belongs to same community.
    """
    try:
        size = len(chrom)
        last_c = 0
        communities = [float("inf")] * size
        pending = set(range(size))

        while len(pending) != 0:
            index = int(pending.pop())
            neighbour = int(chrom[index])

            if neighbour != -1:
                communities[index] = min(last_c, communities[index], communities[neighbour])
                while neighbour in pending:
                    pending.remove(neighbour)
                    communities[neighbour] = min(last_c, communities[neighbour])
                    neighbour = int(chrom[neighbour])
            last_c += 1
        return communities
    except Exception as e:
        raise e


def modularity_individual(individual, graph: igraph.Graph):
    """
    Decode an individual into a community membership vector and
    calculates the modularity of the individual community set.
    """
    members = decode(individual)
    try:
        return graph.modularity(members)
    except igraph.InternalError as e:
        raise RuntimeError("individual: {0}, members:{1}. raise the exception {2}".format(individual, members, e))


def evaluate_individual(individual, graph: igraph.Graph):
    """
    Decode an individual into a community membership vector and
    calculates the modularity of the individual community set.
    """
    members = decode(individual)
    try:
        odf, cs = average_odf_and_community_score(graph, members)
        cs = max(cs, 0.1)
        return odf, 1.0/cs
    except igraph.InternalError as e:
        raise RuntimeError("individual: {0}, members:{1}. raise the exception {2}".format(individual, members, e))


def flip_coin_with_probability(probability: float):
    """
    Returns 0 or 1 with the given probability
    """
    value = 0
    if random.random() <= probability:
        value = 1
    return value


def mutate_individual(individual, graph: igraph.Graph, probability: float):
    """
    Mutate each gen of the individual with the given probability to a random
    neighbor of the node with same id as the gen position
    """
    size = len(individual)

    for i in range(size):
        if flip_coin_with_probability(probability) is 1:
            neighbors = graph.neighborhood(vertices=i)
            individual[i] = neighbors[random.randint(0, len(neighbors) - 1)]

    return individual,


def create_individual(i_type:int, container, graph: igraph.Graph, n: int):
    """
    Creates a random individual creating each gen by selecting a random neighbor of
    the node with same id as the gen position
    """
    ind = container([None] * n)
    ind.i_type = i_type
    ind.n_repro = 0

    for i in range(n):
        neighbors = graph.neighborhood(vertices=i)
        neighbors.remove(i)
        ind[i] = neighbors[random.randint(0, len(neighbors) - 1)]

    return ind


def get_ref_point(g: igraph.Graph) -> Tuple[float, float]:
    n = g.vcount()
    return n, 10

# FITNESS


def order_by_community(members: List[int], nodes_index: List[int]) -> Dict[int, List[int]]:
    aux_comminities = {}

    for index in nodes_index:
        comm_id = members[index]
        if comm_id not in aux_comminities:
            aux_comminities[comm_id] = []
        aux_comminities[comm_id].append(index)

    return aux_comminities


def get_bigger_community(communities_by_nodes: Dict[int, List[int]]) -> List[int]:
    bigger_community = None
    n_members = 0

    for members in communities_by_nodes.values():
        size = len(members)
        if size > n_members:
            bigger_community = members
            n_members = size
    return bigger_community


def find_equivalences_and_members(new_g: igraph.Graph, old_g: igraph.Graph,
                                  old_sol: List[int]) -> (Dict[int, int], Dict[int, int], List[int]):
    equivalence_old_new = {}
    equivalence_new_old = {}
    members = [-1]*new_g.vcount()

    for v in old_g.vs:
        name = v["name"]
        node_new_g = new_g.vs.select(name=name)

        if len(node_new_g) > 0:
            index_new_g = node_new_g[0].index

            equivalence_old_new[v.index] = index_new_g
            equivalence_new_old[index_new_g] = v.index

            members[index_new_g] = old_sol[v.index]

    return equivalence_old_new, equivalence_new_old, members


def random_neighbor(graph: igraph.Graph, vertex_index: int):
    neighbors = graph.neighborhood(vertices=vertex_index)
    return neighbors[random.randint(0, len(neighbors) - 1)]


def average_odf_and_internal_density(graph: igraph.Graph, members: List[int]) -> (float, float):
    """
    Calculates the average odf and the internal density of a solution in one go
    :param graph: graph to calculate the metric to
    :param members: community of each node
    :return: both measures
    """
    v_cluster = igraph.VertexClustering(graph, membership=members)
    edges_crossing = v_cluster.crossing()
    communities = v_cluster.subgraphs()

    n_communities = len(communities)
    average_odf = np.zeros(n_communities)
    internal_density = np.zeros(n_communities)

    comm_sizes = np.array([g.vcount() for g in communities])

    for e_index, crossing in enumerate(edges_crossing):
        if crossing:
            # edge between clusters, influence avg_odf
            edge = graph.es[e_index]

            source_degree = graph.degree(edge.source)
            average_odf[members[edge.source]] += 1.0/source_degree

            target_degree = graph.degree(edge.target)
            average_odf[members[edge.target]] += 1.0 / target_degree
        else:
            # edge in clusters influence internal density
            edge = graph.es[e_index]
            internal_density[members[edge.source]] += 1

    # filter empty communities
    non_empty_comms = np.where(comm_sizes > 1)
    comm_sizes = np.take(comm_sizes, non_empty_comms)
    average_odf = np.take(average_odf, non_empty_comms)
    internal_density = np.take(internal_density, non_empty_comms)

    # aggregate values
    average_odf = average_odf / comm_sizes
    internal_density = (2.0 * internal_density) / (comm_sizes * (comm_sizes - 1.0))
    internal_density = 1.0 - internal_density

    average_odf = np.mean(average_odf)
    internal_density = np.mean(internal_density)

    return average_odf, internal_density


def average_odf_and_community_score(graph: igraph.Graph, members: List[int]) -> Tuple[float, float]:
    """
    Calculates the average odf and the internal density of a solution in one go
    :param graph: graph to calculate the metric to
    :param members: community of each node
    :return: both measures
    """
    v_cluster = igraph.VertexClustering(graph, membership=members)
    edges_crossing = v_cluster.crossing()
    communities = v_cluster.subgraphs()

    n_communities = len(communities)
    average_odf = np.zeros(n_communities)
    community_score = np.zeros(n_communities)

    comm_sizes = np.array([g.vcount() for g in communities])

    for e_index, crossing in enumerate(edges_crossing):
        if crossing:
            # edge between clusters, influence avg_odf
            edge = graph.es[e_index]

            source_degree = graph.degree(edge.source)
            average_odf[members[edge.source]] += 1.0/source_degree

            target_degree = graph.degree(edge.target)
            average_odf[members[edge.target]] += 1.0/target_degree
        else:
            # edge in clusters influence internal density
            edge = graph.es[e_index]
            community_score[members[edge.source]] += 1.0

    # filter empty communities
    non_empty_comms = np.where(comm_sizes >= 1)
    comm_sizes = np.take(comm_sizes, non_empty_comms)
    average_odf = np.take(average_odf, non_empty_comms)
    community_score = np.take(community_score, non_empty_comms)

    # aggregate values
    average_odf = average_odf / comm_sizes
    community_score = ((2.0 * community_score) / comm_sizes)**2

    average_odf = np.sum(average_odf)
    community_score = np.sum(community_score)

    return average_odf, community_score


# REPARATION
def do_convergence_steps(new_individual: creator.Individual, num_iter: int, actual_snapshot: igraph.Graph):
    for i in range(num_iter):
        members = decode(new_individual)

        vertex_order = list(range(actual_snapshot.vcount()))
        random.shuffle(vertex_order)

        for index in vertex_order:
            neighbors = actual_snapshot.neighborhood(vertices=index)
            if len(neighbors) > 1:
                communities_by_nodes = order_by_community(members, neighbors)
                bigger_community = get_bigger_community(communities_by_nodes)

                connected_node = new_individual[index]
                if connected_node not in bigger_community:
                    new_individual[index] = bigger_community[random.randint(0, len(bigger_community) - 1)]
                    members = decode(new_individual)


# TYPE CROSSOVER
def is_immigrant(ind: creator.Individual) -> bool:
    return ind.i_type == IMMIGRANT or ind.i_type == IMMIGRANT_IMMIGRANT


def is_random(ind: creator.Individual) -> bool:
    return ind.i_type == RANDOM or ind.i_type == RANDOM_RANDOM or ind.i_type == IMMIGRANT_RANDOM


def type_crossover(ind1: creator.Individual, ind2: creator.Individual) -> int:

    if ind1.i_type == IMMIGRANT_RANDOM and is_immigrant(ind2) or ind2.i_type == IMMIGRANT_RANDOM and is_immigrant(ind1):
        # r+i + i/i+i = i+i
        return IMMIGRANT_IMMIGRANT
    elif ind1.i_type == IMMIGRANT_RANDOM and is_random(ind2) or ind2.i_type == IMMIGRANT_RANDOM and is_random(ind1):
        # r+i + r/r+r/r+i = r+r
        return RANDOM_RANDOM
    elif is_immigrant(ind1) and is_random(ind2) or is_immigrant(ind2) and is_random(ind1):
        return IMMIGRANT_RANDOM
    elif is_immigrant(ind1) and is_immigrant(ind2):
        # i/i+i + i/i+i = i+i
        return IMMIGRANT_IMMIGRANT
    elif is_random(ind1) and is_random(ind2):
        # r/r+r + r/r+r = r+r
        return RANDOM_RANDOM
    else:
        raise RuntimeError("unknow types combination {0}-{1}".format(ind1.i_type, ind2.i_type))


def count_types(population: List[creator.Individual])-> Tuple[int,int, int,int,int]:
    n_immigrants = 0
    n_immigrants_immigrants = 0
    n_random = 0
    n_random_random = 0
    n_random_immigrants = 0

    for ind in population:
        if ind.i_type == RANDOM:
            n_random += 1
        elif ind.i_type == RANDOM_RANDOM:
            n_random_random += 1
        elif ind.i_type == IMMIGRANT:
            n_immigrants += 1
        elif ind.i_type == IMMIGRANT_IMMIGRANT:
            n_immigrants_immigrants += 1
        elif ind.i_type == IMMIGRANT_RANDOM:
            n_random_immigrants += 1
        else:
            raise RuntimeError("Unknown i_type: {0}".format(ind.i_type))

    return n_immigrants, n_immigrants_immigrants, n_random, n_random_random, n_random_immigrants


def split_random_immigrants(population: List[creator.Individual])-> Tuple[List[creator.Individual], List[creator.Individual]]:
    random_inds = []
    immigrant_inds = []
    for ind in population:
        if is_immigrant(ind):
            immigrant_inds.append(ind)
        else:
            random_inds.append(ind)
    return random_inds, immigrant_inds

# -------------------SELECTION-------------------------------------------------------------


def prevent_incest_selection(pop: List[creator.Individual], n_selected: int, incest_allow_rate: float) -> List[creator.Individual]:
    """
    Tournament selection based on dominance (D) between two individuals, if the two individuals do not interdominate
    the selection is made based on crowding distance (CD). The method prevents two immigrant individuals to mate allowing
    only incest between immigrants with a low probability
    :param pop: list of individuals to select from
    :param n_selected: number of individuals to select
    :param incest_allow_rate: probability of an incest to occur
    :return: list of selected individuals
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    def is_crossover_allowed(mate1, mate2):
        if len(randoms) > 0 and is_immigrant(mate1) and is_immigrant(mate2):
            return random.random() < incest_allow_rate
        else:
            return True

    def select_4(individuals, index):
        mate1 = tourn(individuals[index], individuals[index + 1])
        chosen.append(mate1)

        mate2 = tourn(individuals[index + 2], individuals[index + 3])
        if is_crossover_allowed(mate1, mate2):
            chosen.append(mate2)
        else:
            chosen.append(random.choice(randoms))

    randoms = [ind for ind in pop if not is_immigrant(ind)]

    individuals_1 = random.sample(pop, len(pop))
    individuals_2 = random.sample(pop, len(pop))

    chosen = []
    for i in range(0, n_selected, 4):
        select_4(individuals_1, i)
        select_4(individuals_2, i)

    return chosen


def prevent_incest_selection_v2(pop: List[creator.Individual], n_selected: int, incest_allow_rate: float) -> List[creator.Individual]:
    """
    Alternative version of the prevent incest crossover, when a mating between immigrants is forbidden two random inds
    are selected and a tournament decides which mates with the immigrant
    :param pop: list of individuals to select from
    :param n_selected: number of individuals to select
    :param incest_allow_rate: probability of an incest to occur
    :return: list of selected individuals
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    def is_crossover_allowed(mate1, mate2):
        if len(randoms) > 0 and is_immigrant(mate1) and is_immigrant(mate2):
            return random.random() < incest_allow_rate
        else:
            return True

    def select_4(individuals, index):
        mate1 = tourn(individuals[index], individuals[index + 1])
        chosen.append(mate1)

        mate2 = tourn(individuals[index + 2], individuals[index + 3])
        if is_crossover_allowed(mate1, mate2):
            chosen.append(mate2)
        else:
            r1 = random.choice(randoms)
            r2 = random.choice(randoms)
            chosen.append(tourn(r1, r2))

    randoms = [ind for ind in pop if not is_immigrant(ind)]

    individuals_1 = random.sample(pop, len(pop))
    individuals_2 = random.sample(pop, len(pop))

    chosen = []
    for i in range(0, n_selected, 4):
        select_4(individuals_1, i)
        select_4(individuals_2, i)

    return chosen


def prevent_incest_selection_variable_rate(pop: List[creator.Individual], n_selected: int, min_allow_rate: float,
                                           max_allow_rate: float) -> List[creator.Individual]:
    """
    Alternative version of the prevent incest crossover, when a mating between immigrants is forbidden two random inds
    are selected and a tournament decides which mates with the immigrant. The incest rates changes depending on the
    fraction of random individuals.

    :param pop: list of individuals to select from
    :param n_selected: number of individuals to select
    :param min_allow_rate: minimum probability of an incest to occur
    :param max_allow_rate: maximum probability of an incest to occur
    :return: list of selected individuals
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    def is_crossover_allowed(mate1, mate2):
        if len(randoms) > 0 and is_immigrant(mate1) and is_immigrant(mate2):
            return random.random() < incest_allow_rate
        else:
            return True

    def select_4(individuals, index):
        mate1 = tourn(individuals[index], individuals[index + 1])
        chosen.append(mate1)

        mate2 = tourn(individuals[index + 2], individuals[index + 3])
        if is_crossover_allowed(mate1, mate2):
            chosen.append(mate2)
        else:
            r1 = random.choice(randoms)
            r2 = random.choice(randoms)
            chosen.append(tourn(r1, r2))

    randoms = [ind for ind in pop if not is_immigrant(ind)]

    incest_allow_rate = max(min(len(randoms)/len(pop), max_allow_rate), min_allow_rate)

    individuals_1 = random.sample(pop, len(pop))
    individuals_2 = random.sample(pop, len(pop))

    chosen = []
    for i in range(0, n_selected, 4):
        select_4(individuals_1, i)
        select_4(individuals_2, i)

    return chosen


def round_robin_selection(pop: List[creator.Individual], n_selected: int) -> List[creator.Individual]:
    """
    If possible select half of the populations as normal tournament and the other half mixing immigrants and randoms.
    All the selections are made following a tournament scheme.
    :param pop: list of individuals to select from
    :param n_selected: number of individuals to select
    :return: list of selected individuals
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    randoms = []
    immigrants = []

    for ind in pop:
        if is_immigrant(ind):
            immigrants.append(ind)
        else:
            randoms.append(ind)

    r_size = len(randoms)
    i_size = len(immigrants)

    if r_size > 4 and i_size > 4:
        whole_1 = random.sample(pop, len(pop))
        whole_2 = random.sample(pop, len(pop))

        chosen = []
        for i in range(0, int(n_selected/2), 2):
            assert i_size == len(immigrants), "immigrants torunament change size"

            chosen.append(tourn(whole_1[i], whole_1[i + 1]))
            chosen.append(tourn(whole_2[i], whole_2[i + 1]))

            if i % r_size == 0 or (i+1) % r_size == 0:
                random.shuffle(randoms)

            if i % i_size == 0 or (i+1) % i_size == 0:
                random.shuffle(immigrants)

            chosen.append(tourn(randoms[i % r_size], randoms[(i+1) % r_size]))
            chosen.append(tourn(immigrants[i % i_size], immigrants[(i+1) % i_size]))
    else:
        whole_1 = random.sample(pop, len(pop))
        whole_2 = random.sample(pop, len(pop))

        chosen = []
        for i in range(0, n_selected, 4):
            chosen.append(tourn(whole_1[i], whole_1[i + 1]))
            chosen.append(tourn(whole_1[i + 2], whole_1[i + 3]))
            chosen.append(tourn(whole_2[i], whole_2[i + 1]))
            chosen.append(tourn(whole_2[i + 2], whole_2[i + 3]))

    return chosen


def round_robin_selection_v2(pop: List[creator.Individual], n_selected: int) -> List[creator.Individual]:
    """
    Normal tournament that adds a 20% of random and immigrant crossover
    :param pop: list of individuals to select from
    :param n_selected: number of individuals to select
    :return: list of selected individuals
    """
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    randoms = []
    immigrants = []

    for ind in pop:
        if is_immigrant(ind):
            immigrants.append(ind)
        else:
            randoms.append(ind)

    r_size = len(randoms)
    i_size = len(immigrants)

    whole_1 = random.sample(pop, len(pop))
    whole_2 = random.sample(pop, len(pop))

    chosen = []
    mixing_round = False
    for i in range(0, n_selected, 4):
        chosen.append(tourn(whole_1[i], whole_1[i + 1]))
        chosen.append(tourn(whole_1[i + 2], whole_1[i + 3]))

        if mixing_round:
            # add a random-immigrant

            ni = max(i-2, 0)
            # random + immigrant
            if ni % r_size == 0 or (ni + 1) % r_size == 0:
                random.shuffle(randoms)

            if ni % i_size == 0 or (ni + 1) % i_size == 0:
                random.shuffle(immigrants)

            chosen.append(tourn(randoms[i % r_size], randoms[(i + 1) % r_size]))
            chosen.append(tourn(immigrants[i % i_size], immigrants[(i + 1) % i_size]))
        else:
            # normal tournament
            chosen.append(tourn(whole_2[i], whole_2[i + 1]))
            chosen.append(tourn(whole_2[i + 2], whole_2[i + 3]))

        mixing_round = not mixing_round and (r_size > 4 and i_size > 4)

    return chosen


def sel_spread(population: List[creator.Individual], n_selected: int) -> List[creator.Individual]:
    fit_x, fit_y = zip(*[ind.fitness.values for ind in population])
    index = np.lexsort((fit_x, fit_y))

    return [population[index[i]] for i in range(0, len(index), int(math.ceil(len(index)/n_selected)))]


def sel_best_split(population: List[creator.Individual], n_selected: int) -> Tuple[List[creator.Individual], List[creator.Individual]]:
    fit_x, fit_y = zip(*[ind.fitness.values for ind in population])
    index = np.lexsort((fit_y, fit_x))

    return [population[i] for i in index[:n_selected]], [population[i] for i in index[n_selected:]]


def sel_spread_split(population: List[creator.Individual], n_selected: int) -> Tuple[List[creator.Individual], List[creator.Individual]]:
    fit_x, fit_y = zip(*[ind.fitness.values for ind in population])
    index = np.lexsort((fit_y, fit_x))

    range_selected = range(0, len(index), int(math.ceil(len(index)/n_selected)))

    p_selected = []
    p_not_selected = []
    for i in range(len(index)):
        if i in range_selected:
            p_selected.append(population[index[i]])
        else:
            p_not_selected.append(population[index[i]])

    return p_selected, p_not_selected

# ------------------- DOMINANCE -----------------------------------------------------------------------


def maintain_random_dominance(poplation: List[creator.Individual], k: int, min_random_rate: float) -> List[creator.Individual]:
    """ Apply NSGA-II selection operator on the *individuals* like deap does. Mantains a minimum of *min_random_rate* of
        the population as random individuals.

        :param poplation: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param min_random_rate: percentage of the population that will be random individuals
        :returns: A list of selected individuals.
    """

    max_immigrant_pop = int(math.floor((1.0 - min_random_rate)*k))
    min_random_popuation = k - max_immigrant_pop

    pareto_fronts = sortNondominated(poplation, k)
    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = []
    randoms_selected = set()
    n_i = 0
    n_r = 0

    i = 0
    flatten_fronts = list(chain(*pareto_fronts[:-1]))
    while n_i < max_immigrant_pop and i < len(flatten_fronts):
        ind = flatten_fronts[i]
        if is_immigrant(ind):
            n_i += 1
        else:
            n_r += 1
            randoms_selected.add(str(ind))
        chosen.append(ind)
        i += 1

    if n_i < max_immigrant_pop:
        # i have processed all fronts and the maximum number of immigrants have not bee surpassed
        to_add = k - len(chosen)
        if to_add > 0:
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)

            # count the random individuals in the selection, to check if the constraints are met
            r_inds, i_inds = split_random_immigrants(sorted_front[:to_add])
            n_r += len(r_inds)

            if n_r >= min_random_popuation:
                # constraints are met, fill the population
                chosen.extend(sorted_front[:to_add])
            else:
                # not enough randoms add more
                randoms_to_add = min_random_popuation - n_r
                chosen.extend(i_inds[0:-randoms_to_add])

                for ind in r_inds:
                    chosen.append(ind)
                    randoms_selected.add(str(ind))

                randoms = [ind for ind in sorted_front if is_random(ind) and str(ind) not in randoms_selected]
                if len(randoms) >= randoms_to_add:
                    chosen.extend(randoms[:randoms_to_add])
                else:
                    randoms = [ind for ind in poplation if is_random(ind) and str(ind) not in randoms_selected]
                    assert len(randoms) >= randoms_to_add, "not enough randoms"
                    chosen.extend(selNSGA2(randoms, randoms_to_add))
    else:
        # maximum number of immigrants have been surpassed finish with randoms
        randoms = [ind for ind in poplation if is_random(ind) and str(ind) not in randoms_selected]
        chosen.extend(selNSGA2(randoms, min_random_popuation - n_r))

    return chosen



























