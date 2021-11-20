import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import activity, behavior, rates, graphs
from scripting.rett_memory.tools import plotting, pdtools, signals

DATAPATH = paths.data.joinpath('signals_df.pkl')
BEHAVIORPATH = paths.data.joinpath('behavior_df.pkl')
RASTER_EXPS = [('wt','N083', 'NA'), ('het', 'N014', 'NA')]
RASTER_CXTS = ['Fear_2'] * 2

ANIMALPATH = paths.data.joinpath('P80_animals.pkl')
INDEXES = [('wt',),('het',)]
CONTEXTS = ['Neutral', 'Fear', 'Fear_2']


def rasters(exps=RASTER_EXPS, cxts=RASTER_CXTS, df=None, bdf=None):
    """Constructs the raseters and network activity traces of panels A and
    B for Figure 3 of the paper."""

    #scale amplitude of network activity by a multiple to make easier to see
    gain = 3
    #set number of cells in raster
    ncells = 25
    #read in dataframes of signals and behaviors
    df = pd.read_pickle(DATAPATH) if df is None else df
    df = df.sort_index()
    bdf = pd.read_pickle(BEHAVIORPATH) if bdf is None else bdf
    #compute the networ activity traces
    activities = activity.NetworkActivity(df, std=5).measure()
    fig, axarr = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
    for idx, (exp, cxt) in enumerate(zip(exps, cxts)):
        spike_indices = df.loc[exp][cxt + '_spikes']
        
        #grab the first 15 cells spike indices
        for row, subls in enumerate(spike_indices[:ncells]):
            spike_times = np.array(subls) / df.iloc[0].sample_rate
            axarr[idx].scatter(spike_times, row * np.ones(len(spike_times)),
                               marker='|', s=5, color='k')
        trace = activities.loc[exp][cxt]
        #place activity trace at top of plot and scale it
        axarr[idx].plot(np.linspace(0, 300, 6000), gain * trace + ncells+1)
        
        #get the freezing signal and threshold to remove freezes < 2 secs
        btimes = np.array(bdf.loc[exp][cxt + '_time'])
        freezes = behavior.threshold(bdf.loc[exp][cxt + '_freeze'], 
                                     btimes, 2)
        #build rectangles of freezing times
        bottom, top = axarr[idx].get_ylim()
        height = top - bottom
        crosses = signals.crossings(freezes, level=1).astype(int)
        cross_times = btimes[crosses.flatten()].reshape(-1,2)
        rects = [Rectangle((c[0], bottom), width=c[1]-c[0], height=height,
                 facecolor='gray', alpha=0.25) for c in cross_times]
        [axarr[idx].add_patch(rect) for rect in rects]
        #add scalebar 1 unit = gain HZ
        axarr[idx].plot([10,10], [ncells + 2, ncells + 2 + gain],
                        color='pink')

    #compute the network activity thresholds
    measurer = activity.NetworkBursts(df)
    #compute the activity traces for each cell using a width of 5
    cell_activities, _ = measurer.activity(5)
    thresholds = measurer.threshold(cell_activities, shifts=[500, 1500],
                                    repeats=10, nstds=1.5)
    heights = [thresholds.loc[e][c] for e,c in zip(exps, cxts)]
    [axarr[i].axhline(gain*y + ncells+1, color='r', linestyle='--') for i, y in
            enumerate(heights)]

    fig.tight_layout()
    plt.show()

def coactivity_boxplot(groups=INDEXES, categories=CONTEXTS, 
                       ylabel='Coactive Cell Percentage', showfliers=False):
    """Constructs a boxplot of the percentage of coactive cells during the
    peak of the network activity trace for figure 3C of the paper."""

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #make the coactivity metric
    metric = activity.Coactivity(data)
    percentages = metric.measure(band=20, level=1, width=5, 
                                 shifts=[500, 1500], repeats=10, nstds=1.5)
    #percentages = percentages.fillna(0)
    result_dict = pdtools.df_to_dict(percentages, groups, categories)
    #make categorical boxplot
    plotting.boxplot(result_dict, categories, groups, ylabel=ylabel,
                     showfliers=showfliers)
    return percentages

def coactivity_rates_boxplot(groups=[('wt',), ('het',),
                                     ('wt', True), ('het', True),
                                     ('wt', False),('het', False)],
                             categories=['Fear_2'], 
                             ylabel='Event Rates', showfliers=False):
    """Boxplots the rates for all cells, coactive cells and non-coactive 
    cells."""

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #compute the spike rates
    spike_rates = rates.Rate(data).measure()
    #identify cells as co-active or non co-active
    cocells = activity.CoactiveCells(data).measure().astype(bool)
    #get the rates of the co/non-co active cells
    corates = spike_rates.loc[cocells.Fear_2]
    noncorates = spike_rates.loc[~cocells.Fear_2]
    #groupby geno and mouse id only (drop treatment, cell id)
    group_idx = data.index.names[:-2]
    mean_corates = corates.groupby(by=group_idx).apply(np.mean)
    mean_noncorates = noncorates.groupby(by=group_idx).apply(np.mean)
    #add coactive column to each df
    mean_corates = mean_corates.assign(coactive=np.ones(len(mean_corates),
                                       dtype=bool))
    mean_noncorates = mean_noncorates.assign(coactive=np.zeros(
                                    len(mean_noncorates), dtype=bool))
    #move the coactive column to an index
    mean_corates = mean_corates.set_index('coactive', append=True)
    mean_noncorates = mean_noncorates.set_index('coactive', append=True)
    #join the dataframes
    mean_rates = pd.concat([mean_corates, mean_noncorates])
    #convert the rates to a dict for plotting
    g = groups[2:] #exclude the merged rates
    result_dict = pdtools.df_to_dict(mean_rates, g, categories)
    # compute the merged spike rates
    all_rates = rates.Rate(data).measure(cell_avg=True)
    all_rates_dict = pdtools.df_to_dict(all_rates, categories=categories,
                                   groups=groups[:2])
    #update the results dict with the merged rates
    result_dict.update(all_rates_dict)
    #make categorical boxplot
    plotting.boxplot(result_dict, categories, groups, ylabel=ylabel,
                     showfliers=showfliers)
    return mean_rates, all_rates


def sample_graphs(exps=[('wt', 'N087', 'NA'), ('het', 'N229', 'NA'),
                        ('wt','N008', 'NA'), ('het','N014','NA'),
                        ('wt','N083','NA'),('het','sn221','NA')]):
    """Plots sample graphs for 3 WT and 3 RTT mice for panel F of Fig 3."""



    """
    exps=[('wt', 'N087', 'NA'),('het', 'N229', 'NA'),
                    ('wt', 'N008', 'NA'), ('het','N014','NA'),
                    ('wt','N083','NA'),('het','N075','NA'),
                    ('wt','N087','NA'),('het','sn221','NA'),
                    ('wt','N019','NA'),('het','N075','NA')]
    """

    #read in needed dataframes
    pairs_df = pd.read_pickle(paths.data.joinpath('correlated_pairs_df.pkl'))
    hd_df = pd.read_pickle(paths.data.joinpath('high_degree_df.pkl'))
    rois_df = pd.read_pickle(paths.data.joinpath('rois_df.pkl'))

    fig, axarr = plt.subplots(len(exps), 2, sharex=True, sharey=True)
    for row, exp in enumerate(exps):
        for col, cxt in enumerate(('Neutral', 'Fear_2')):
            pairs = pairs_df.loc[exp][cxt]
            hds = hd_df.loc[exp][cxt]
            #create the graph
            G = nx.Graph()
            G.add_edges_from(pairs[:,:2])
            #get the high degree subgraph
            g = graphs.subgraph(G, hds)
            centroids = rois_df.loc[exp]['centroid'][hds]
            pos = dict(zip(hds, centroids))
            #convert px pos to um positions
            pos = {k: (pos[0]/0.83, pos[1]/0.8) for k, pos in pos.items()}
            color = 'tab:blue' if 'wt' in exp else 'tab:orange'
            graphs.draw(g, pos=pos, with_labels=False,
                                  node_color=color, edge_color='tab:gray', 
                                  node_size=30, ax=axarr[row, col],
                                  width=0.25)
            axarr[row, col].tick_params(left=True, bottom=True, 
                                        labelleft=True, labelbottom=True)
    plt.show()

def ensemble_size_boxplot(groups=INDEXES, categories=CONTEXTS, **kwargs):
    """Constructs a boxplot of the number of High-degree cells in each group
    and context.

    Args:
        groups (list):              list of dataframe keys to display
        categories (list):          list of context names to display on plot

    Returns: dataframe of counts
    """

    hd_df = pd.read_pickle(paths.data.joinpath('high_degree_df.pkl'))
    #count the cells for each exp and context
    results_df = hd_df.applymap(len)
    #compute dict keyed on groups with np.arr values one col per cat.
    results = pdtools.df_to_dict(results_df, groups, categories)
    plotting.boxplot(results, categories, groups, **kwargs)
    return results_df

def enemble_degree_boxplot(groups=INDEXES, categories=CONTEXTS, **kwargs):
    """Constructs a boxplot of the average degree of the high degree cells.
    
    Args:
        groups (list):              list of dataframe keys to display
        categories (list):          list of context names to display on plot
    
    Returns: dataframe of degrees
    """

    #read in needed dataframes
    pairs_df = pd.read_pickle(paths.data.joinpath('correlated_pairs_df.pkl'))
    hd_df = pd.read_pickle(paths.data.joinpath('high_degree_df.pkl'))

    def avg_deg(arr, hd_df):
        """Returns average degree of high-degree cells in a pairs array."""
        
        #make graph of pairs and extract high-degree subgraph
        G = graphs.make_graph(arr[:,:2])
        g = graphs.subgraph(G, hd_df)
        return graphs.avg_degree(g)
    
    result = dict()
    contexts = pairs_df.columns
    for ntup in pairs_df.itertuples():
        result[ntup.Index] = {cxt: avg_deg(getattr(ntup, cxt), 
                                           hd_df.loc[ntup.Index][cxt]) 
                              for cxt in contexts}
    results_df = pd.DataFrame.from_dict(result, orient='index')
    results_df.index.names = pairs_df.index.names
    #compute dict keyed on groups with np.arr values one col per cat.
    results = pdtools.df_to_dict(results_df, groups, categories)
    plotting.boxplot(results, categories, groups, **kwargs)
    return results_df

def high_degree_rates(groups=None, categories=None, **kwargs):
    """ """

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #compute the spike rates of all cells
    spike_rates = rates.Rate(data).measure()
    #read in the high_degree cells df
    hd_df = pd.read_pickle(paths.data.joinpath('high_degree_df.pkl'))
    
    #make a boolean df filled with False & same index as spike_rates
    hd_bool_df = pd.DataFrame(0, spike_rates.index, 
                              spike_rates.columns).astype(bool)
    #set the hd_bool df to True where there is a high_degree cell
    for context in hd_bool_df.columns:
        for exp in hd_bool_df.index:
            #if cell is in hd_df at this exp and context -- set to True
            if exp[-1] in hd_df.loc[exp[:-1]][context]:
                hd_bool_df.loc[exp][context] = True

    #fetch hd, non-hd and all rates in Fear_2            
    hd_rates = spike_rates.loc[hd_bool_df.Fear_2]
    non_hd_rates = spike_rates.loc[~hd_bool_df.Fear_2]
    all_rates = spike_rates.Fear_2

    #add a new index called HD that designates if cell is high-degree
    hd_rates = hd_rates.assign(HD = np.ones(len(hd_rates),dtype=bool))
    non_hd_rates = non_hd_rates.assign(HD = np.zeros(len(non_hd_rates)),
                                       dtype=bool)
    hd_rates = hd_rates.set_index('HD', append=True)
    non_hd_rates = non_hd_rates.set_index('HD', append=False)
    return hd_rates, non_hd_rates
    
    










if __name__ == '__main__':

    from scripting.rett_memory.tools import stats
    plt.ion()

    """#Fig 3 A-B
    rasters()
    """

    """#Fig 3C
    results = coactivity_boxplot()
    s = stats.row_compare(results)
    """
   
    """#Fig 3D
    split_rates, all_rates = coactivity_rates_boxplot()
    s = stats.row_compare(results)
    """

    """#Fig 3F
    sample_graphs()
    """

    """#Fig 3G
    results = ensemble_size_boxplot(showfliers=False)
    s = stats.column_compare(results)
    """

    """#Fig 3H
    results = enemble_degree_boxplot(showfliers=False)
    s  =stats.column_compare(results)
    """

    hd_rates = high_degree_rates()
