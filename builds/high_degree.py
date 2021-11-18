import pandas as pd
import networkx as nx

from scripting.rett_memory.metrics import graphs

def high_degree(pairs, repeat=100, std_factor=1.5):
    """Returns a dataframe of lists of cells whose degrees exceeds a
    randomly connected graph of the same node and edge count.
    
    Args:
        pairs (dataframe):      dataframe of arrays, with each array
                                row containing a pair and strength
        repeat (int):           num times to generate a random graph
                                to find high degree pairs (see
                                graphs.random_degree; Default=100)
        std_factor (int):       num of stds of max degrees of random
                                graphs beyond which a node is considered
                                high degree (Default is 1.5 std)
    """

    def hds(pair_arr):
        """Returns a list of cells from an array of pairs whose degree
        exceeds a random graph's max degrees."""
        
        #build graph and compute random degree statistics
        G = nx.Graph()
        G.add_edges_from(pair_arr[:,:2])
        avg_max, std_max = graphs.random_degree(G, repeat)
        #threshold pairs returning cells whose degree exceeds random
        threshold = avg_max + std_factor * std_max
        return [int(cell) for cell, deg in G.degree if deg > threshold]

    return pairs.applymap(hds)

if __name__ == '__main__':

    from scripting.rett_memory import paths

    pairs = pd.read_pickle(paths.data.joinpath('correlated_pairs_df.pkl'))
    hd_cells = high_degree(pairs)
    hd_cells.to_pickle(paths.data.joinpath('high_degree_df.pkl'))
