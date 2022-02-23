import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


from scripting.rett_memory import paths
from scripting.rett_memory.metrics import rates, activity, behavior
from scripting.rett_memory.tools import plotting, pdtools, stats, signals

DATAPATH = paths.data.joinpath('signals_df.pkl')
ANIMALPATH = paths.data.joinpath('P80_animals.pkl')
BEHAVIORPATH = paths.data.joinpath('behavior_df.pkl')
INDEXES = [('wt',),('het',)]
#CONTEXTS = ['Neutral', 'Neutral_2']
CONTEXTS = ['Neutral_2', 'Fear', 'Fear_2']
#RASTER_EXPS = [('wt','N124', 'NA'), ('het', 'N229', 'NA')]
RASTER_EXPS = [('wt','N083', 'NA'), ('het', 'N014', 'NA')]
RASTER_CXTS = ['Fear_2'] * 2

def rates_boxplot(groups=INDEXES, categories=CONTEXTS, cell_avg=True, 
                  ylabel='Event Rate', showfliers=False):
    """Constructs a boxplot of spike rates for Supplemental Figure 3.

    Returns: dataframe of rates
    """

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #measure the spike rates and convert to dict for plotting
    measurer = rates.Rate(data)
    result_df = measurer.measure(cell_avg=cell_avg)
    result_dict = pdtools.df_to_dict(result_df, groups, categories)
    #make categorical boxplot
    plotting.boxplot(result_dict, categories, groups, ylabel=ylabel,
                     showfliers=showfliers)
    return result_df

def rasters(exps=RASTER_EXPS, cxts=RASTER_CXTS):
    """Constructs the raseters and network activity traces of panels A and
    B for Figure 3 of the paper and supplemental Figure 3D."""

    #scale amplitude of network activity by a multiple to make easier to see
    gain = 10
    #read in dataframes of signals and behaviors
    df = pd.read_pickle(DATAPATH)
    df = df.sort_index()
    bdf = pd.read_pickle(BEHAVIORPATH)
    #compute the networ activity traces
    activities = activity.NetworkActivity(df, std=5).measure()
    fig, axarr = plt.subplots(1, 2, figsize=(8,6), sharex=True,
            sharey=True)
    ncells = []
    for idx, (exp, cxt) in enumerate(zip(exps, cxts)):
        spike_indices = df.loc[exp][cxt + '_spikes']
        ncells.append(len(spike_indices)) 
        #grab the first 15 cells spike indices
        for row, subls in enumerate(spike_indices):
            spike_times = np.array(subls) / df.iloc[0].sample_rate
            axarr[idx].scatter(spike_times, row * np.ones(len(spike_times)),
                               marker='|', s=5, color='k')
        trace = activities.loc[exp][cxt]
        #place activity trace at top of plot and scale it
        axarr[idx].plot(np.linspace(0, 300, 6000), 
                        gain * trace + ncells[idx]+1)
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
        #add scalebar 10 units = 1 HZ (+2 for a little offset from raster)
        axarr[idx].plot([10,10], [ncells[idx] + 2, (ncells[idx] + 2) + gain])
    #compute the network activity thresholds
    measurer = activity.NetworkBursts(df)
    #compute the activity traces for each cell using a width of 5
    cell_activities, _ = measurer.activity(5)
    thresholds = measurer.threshold(cell_activities, shifts=[500, 1500],
                                    repeats=10, nstds=1.5)
    heights = [thresholds.loc[e][c] for e,c in zip(exps, cxts)]
    [axarr[i].axhline(gain*y + ncells[i] + 1, color='r', linestyle='--') 
                      for i, y in enumerate(heights)]
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    """
    plt.ion()
    results = rates_boxplot()
    stat_results = stats.column_compare(results)

    print(results.columns)
    print('wt medians')
    print(np.median(results.loc[('wt',)], axis=0))
    print('wt IQRS')
    print(np.percentile(results.loc[('wt',)], q=[25,75], axis=0))

    print('rtt medians')
    print(np.median(results.loc[('het',)], axis=0))
    print('rett IQRS')
    print(np.percentile(results.loc[('het',)], q=[25,75], axis=0))
    """

    #Supp Fig 3D
    rasters()
