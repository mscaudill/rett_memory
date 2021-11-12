import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import activity, behavior
from scripting.rett_memory.tools import plotting, pdtools

DATAPATH = paths.data.joinpath('signals_df.pkl')
BEHAVIORPATH = paths.data.joinpath('behavior_df.pkl')
RASTER_EXPS = [('wt','N124', 'NA'), ('het', 'N229', 'NA')]
RASTER_CXTS = ['Fear_2'] * 2


def rasters(exps=RASTER_EXPS, cxts=RASTER_CXTS):
    """Constructs the raseters and network activity traces of panels A and
    B for Figure 3 of the paper."""

    #read in dataframes of signals and behaviors
    df = pd.read_pickle(DATAPATH)
    df = df.sort_index()
    bdf = pd.read_pickle(BEHAVIORPATH)
    print('data loaded')
    activities = activity.NetworkActivity(df, std=5).measure()
    fig, axarr = plt.subplots(2, 2, figsize=(8,3), sharex=True)
    for idx, (exp, cxt) in enumerate(zip(exps, cxts)):
        #get plot position coords
        pos = np.unravel_index(idx, axarr.shape)
        spike_indices = df.loc[exp][cxt + '_spikes']
        #grab the first 15 cells spike indices
        for row, subls in enumerate(spike_indices[0:15]):
            spike_times = np.array(subls) / df.iloc[0].sample_rate
            axarr[pos].scatter(spike_times, row * np.ones(len(spike_times)),
                               marker='|', s=5, color='k')
        trace = activities.loc[exp][cxt]
        #place activity trace at top of plot and scale it
        axarr[pos].plot(np.linspace(0, 300, 6000), 3 * trace + 16)
        #get the freezing signal and threshold to remove freezes < 2 secs
        btimes = bdf.loc[exp][cxt + '_time']
        freezes = behavior.threshold(bdf.loc[exp][cxt + '_freeze'], 
                                     btimes, 2)
        axarr[1, idx].plot(btimes, freezes)
    #compute the network activity thresholds
    print('computing thresholds')
    measurer = activity.NetworkBursts(df)
    #compute the activity traces for each cell using a width of 5
    cell_activities, _ = measurer.activity(5)
    thresholds = measurer.threshold(cell_activities, shifts=[500, 1500],
                                    repeats=10, nstds=1.5)
    heights = [thresholds.loc[e][c] for e,c in zip(exps, cxts)]
    [axarr[0, i].axhline(3*y + 16, color='r', linestyle='--') for i, y in
            enumerate(heights)]

    # FIXME 
    # ADD SCALEBAR

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    rasters()
