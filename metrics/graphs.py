import numpy as np
import networkx as nx

from operator import itemgetter

def make_graph(edges):
    """Constructs a graph from a set of edges."""

    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def subgraph(g, nodes):
    """Returns a subgraph of g containing nodes."""

    return g.subgraph(nodes)

def draw(g, **kwargs):
    """Draws the graph g using one of the simple builtin algorithms of
    networkx. For more control see networkx.draw_networkx(). """

    nx.draw_networkx(g, **kwargs)

def num_edges(g, weight=None):
    """Returns the size of graph g as the number of edges or total of edge
    weights.

    g (graph):      a networkx graph instance
    weight (str):   edge attribute name holding weight numerical value.

    See networkx Graph.size.
    """

    return g.number_of_edges()

def num_nodes(g):
    """Returns the number of nodes of graph g."""

    return g.number_of_nodes()

def degrees(g):
    """Returns the degree for each node in the network."""

    return nx.degree(g)

def avg_degree(g):
    """Returns the average degree of graph g."""

    return np.mean([deg for node, deg in degrees(g)])

def max_degree(g):
    """Returns the maximum degree and corresponding node of g.
    
    If multiple nodes share the maximum degree the first occurence is
    returned.
    """
    
    return max([(node, deg) for node, deg in degrees(g)], key=itemgetter(1))

def random_degree(g, repeat):
    """Returns the average (and std) maximum degree of repeat number of 
    randomly connected graphs with the same number of nodes and edges as g.
    
    random_degree defines the maximum degree to expect of a graph with n
    nodes and m edges. Nodes in g with degree greater than this number are
    considered high degree nodes.
    """

    n, m = num_nodes(g), num_edges(g)
    observed = []
    for _ in range(repeat):
        gnm = nx.gnm_random_graph(n, m)
        node, deg = max_degree(gnm)
        observed.append(deg)
    return np.mean(observed), np.std(observed)

def num_components(g):
    """Returns the number of connected components of g."""

    return nx.number_connected_components(g)

def largest_component(g):
    """Returns the largest connected component of graph g."""

    return max(nx.connected_components(g), key=len) if g else []

def density(g, min_nodes):
    """Returns the density of g ignoring self loops."""

    N = num_nodes(g)
    E = num_edges(g)

    if N < min_nodes:
        return np.NaN

    if N <= 2:
        return 1
    #max num edges for undirected is N choose 2 and directed is 1/2 that
    E_max = N*(N-1) if g.is_directed() else N*(N-1) / 2
    #get self loops
    nloops = nx.number_of_selfloops(g)
    #computing density ignoring all self loops
    density = (E - nloops) / (E_max)
    return density

def find_cliques(g):
    """Returns all the maximal cliques in a graph g.

    A maximal clique is a complete subgraph of g such that every node is
    adjacent to all other nodes in the subgraph "All-to-All"."""

    return nx.find_cliques(g)

def num_cliques(g, max_density, cliques=None):
    """Returns the number of maximal in cliques in graph g.

    Args:
        cliques (iterator):         iterator of cliques in g if known
        max_density (float):        float in (0, 1] density at which graph
                                    is considered complete (i.e. one clique)

    Return (int): number of cliques

    Notes: If the density of the graph exceeds the max density we consider
    the graph to be complete. This avoids the exponential runtime
    performance of the clique finding problem but it should be noted that
    for these graphs we may miss some cliques.
    """

    #if the graph is nearly complete return a single-clique
    if density(g) >= max_density:
        return 1
    elif not cliques:
        cliques = find_cliques(g)
        #count the cliques
        count = sum(1 for _ in cliques)
        return count

def clique_number(g, max_density, cliques=None):
    """Returns the largest clique size of all maximal cliques in g.
    
    Args:
        cliques (iterator):         iterator of cliques in g if known
        max_density (float):        float in (0, 1] density at which graph
                                    is considered complete (i.e. one clique)

    Return (int) size of largest clique

    Please see notes on nun_cliques
    """
    
    #if the graph is nearly complete return number of nodes of g
    if density(g) >= max_density:
        return len(g.nodes)
    elif not cliques:
        cliques = find_cliques(g)
        return nx.graph_clique_number(g, cliques=cliques)

def clustering_coeff(g):
    """Returns the average clustering coeffecient of all nodes in g.
    
    Note: this is the number of closed triplets divided by number of
    possible triplets.
    """

    return nx.algorithms.average_clustering(g) if g else np.NAN

def avg_shortest_path(g):
    """Returns the average shortest path length between all the nodes.

    Note raises Networkx error if the graph g is not connected."""

    return nx.algorithms.average_shortest_path_length(g)

def algebraic_connectivity(g, **kwargs):
    """Returns the algebraic connectivity of graph g. """

    if num_nodes(g) < 2:
        return np.NAN
    else:
        return nx.linalg.algebraic_connectivity(g, **kwargs)

def main_core(g):
    """Returns the main-core of the k-core decomposition.

    The main core is the k-core with largest core number. The core number of
    a node is the largest value of k of a k-core containing that node."""

    return nx.algorithms.core.k_core(g)

