import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import activity, behavior
from scripting.rett_memory.tools import plotting, pdtools, signals

DATAPATH = paths.data.joinpath('signals_df.pkl')
BEHAVIORPATH = paths.data.joinpath('behavior_df.pkl')
RASTER_EXPS = [('wt','N124', 'NA'), ('het', 'N229', 'NA')]
#RASTER_EXPS = [('wt','N062', 'NA'), ('het', 'N229', 'NA')]
RASTER_CXTS = ['Fear_2'] * 2


def rasters(exps=RASTER_EXPS, cxts=RASTER_CXTS, df=None, bdf=None):
    """Constructs the raseters and network activity traces of panels A and
    B for Figure 3 of the paper."""

    #scale amplitude of network activity by a multiple to make easier to see
    gain = 3
    #set number of cells in raster
    ncells = 15
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
        for row, subls in enumerate(spike_indices[0:ncells]):
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

    #compute the network activity thresholds
    print('computing thresholds')
    measurer = activity.NetworkBursts(df)
    #compute the activity traces for each cell using a width of 5
    cell_activities, _ = measurer.activity(5)
    thresholds = measurer.threshold(cell_activities, shifts=[500, 1500],
                                    repeats=10, nstds=1.5)
    heights = [thresholds.loc[e][c] for e,c in zip(exps, cxts)]
    [axarr[i].axhline(gain*y + ncells+1, color='r', linestyle='--') for i, y in
            enumerate(heights)]
    #add scalebar 1 unit = gain HZ
    axarr[0].plot([10,10], [ncells, (ncells+2) + gain])

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    rasters()
