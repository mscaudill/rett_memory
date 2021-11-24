import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import behavior
from scripting.rett_memory.tools import plotting

BASE = Path('/media/matt/Zeus/Lingjie/data/N019_wt_female_mcorrected/')

ANIMALS = 'P80_animals.pkl'
INDEXES = [('wt',),('het',)]
CONTEXTS = ['Train', 'Neutral'] #or ['Fear', 'Fear_2']

#Raw image and MIP panel
MOUSE = ('wt', 'N019')
IMG_NUM = 4828

#Signals panel
CONTEXT = 'Neutral'
CELLS = [86, 6, 14, 20, 28, 36, 45, 69, 5, 129]


def freezing_boxplot(df, indexes=INDEXES, contexts=CONTEXTS):
    """Displays a boxplot of the freezing percentage for row label indexes
    and column label contexts.

    Args:
        df (pd.DataFrame):          behavior df containing all behavioral
                                    data ('behavior_df.pkl')
        indexes (seq):              seq. of dataframe row label indexers
                                    (e.g. ('wt',) or ('sst-cre', 'mcherry'))
        contexts (list):            seq of context names to display on plot
                                    (these are column prefixes of df, eg.
                                    'Train', or 'Cue')
    """

    with open(paths.data.joinpath(ANIMALS), 'rb') as infile:
            animals = pickle.load(infile)
    df = df.loc[animals]
    #calculate the freeze percentages for each group and category
    percentages = behavior.freeze_percents(df, contexts, 1)
    #create a dict keyed on group with num_exps X num_cxt array vals
    results = dict()
    for group in indexes:
        #get all category dicts for this group
        dics = [cxt_dic for exp, cxt_dic in percentages.items() 
                if set(group).issubset(set(exp))]
        group_results = []
        #for each category gather all percentages into a list
        for cxt in contexts:
            group_results.append([dic[cxt] for dic in dics])
        #convert to numpy array num_animals X num_contexts
        group_results = np.array(group_results).T
        #save under this group key
        results[group] = group_results
    #call to categorical bar plot
    plotting.boxplot(results, contexts, indexes, ylabel='Freezing %',
                     showfliers=False)
    return results

def plot_mip_sources():
    """Plot max intensity projection of source images for mouse N019_wt."""

    data = np.load(paths.data.joinpath('N019_wt_sources.npy'))
    mip = np.max(data, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mip, cmap='gray')
    plt.show()

def plot_signals(cells=CELLS, context=CONTEXT):
    """Plots signal for each cell in cells for context """

    df = pd.read_pickle(paths.data.joinpath('signals_df.pkl'))
    signals = df.loc[MOUSE][context + '_signals']
    signals = np.array([row for row in signals])[cells,:]
    fig, ax = plt.subplots(figsize=(8, 4))
    time = np.linspace(0, 300, 6000)
    for idx, signal in enumerate(signals):
        ax.plot(time, signal+0.5*idx)
    ax.plot([305,305], [-.1, 0.2], color='k') 
    ax.plot([295, 305], [-.1, -.1], color='k')
    plt.show()

def plot_spikes(cell=CELLS[0], context=CONTEXT):
    """Plots the signal and inferred spike for cell."""

    df = pd.read_pickle(paths.data.joinpath('signals_df.pkl'))
    signal = df.loc[MOUSE][context + '_signals'][cell]
    spikes = df.loc[MOUSE][context + '_spikes'][cell]
    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(signal)
    ax.scatter(spikes, -0.1 * np.ones(len(spikes)), marker='|', s=60,
               color='k')
    plt.show()
    return df


if __name__ == '__main__':

    from scripting.rett_memory.tools.stats import mwu
    plt.ion()

    path = paths.data.joinpath('behavior_df.pkl')
    bdf = pd.read_pickle(path)

    #Fig 1B
    recall_percentages = freezing_boxplot(bdf, contexts=['Fear','Fear_2'])
    inset_percentages = freezing_boxplot(bdf, contexts=['Train','Neutral'])

    print('WT Recall medians')
    print(np.nanmedian(recall_percentages[('wt',)], axis=0))
    print('WT Recall IQRS')
    print(np.nanpercentile(recall_percentages[('wt',)], q=[25, 75], axis=0))
    print('WT Recall Counts\n', recall_percentages[('wt',)].shape[0])

    print('_________')
    print('RTT Recall medians')
    print(np.nanmedian(recall_percentages[('het',)], axis=0))
    print('RTT Recall IQRS')
    print(np.nanpercentile(recall_percentages[('het',)], q=[25, 75], axis=0))
    print('RTT Recall Counts\n', recall_percentages[('het',)].shape[0])

    print('_________')
    print('WT Inset medians')
    print(np.nanmedian(inset_percentages[('wt',)], axis=0))
    print('WT Inset IQRS')
    print(np.nanpercentile(inset_percentages[('wt',)], q=[25, 75], axis=0))
    print('WT Inset Counts\n', inset_percentages[('wt',)].shape[0])

    print('_________')
    print('RTT Inset medians')
    print(np.nanmedian(inset_percentages[('het',)], axis=0))
    print('RTT Inset IQRS')
    print(np.nanpercentile(inset_percentages[('het',)], q=[25, 75], axis=0))
    print('RTT Inset Counts\n', inset_percentages[('het',)].shape[0])

    print('_________')
    #mann-whitney for recall contexts
    U_Fear_1, _ = mwu(recall_percentages[('wt',)][:,0],
                              recall_percentages[('het',)][:,0])
    U_Fear_2, _ = mwu(recall_percentages[('wt',)][:,1],
                              recall_percentages[('het',)][:,1])
    #mann-whitney for train and neutral contexts
    U_Train, _ = mwu(inset_percentages[('wt',)][:,0],
                              inset_percentages[('het',)][:,0])
    U_Neutral, _ = mwu(inset_percentages[('wt',)][:,1],
                              inset_percentages[('het',)][:,1])
    print(U_Fear_1, U_Fear_2, U_Train, U_Neutral)
    #use table since asymptotic normal does not apply for n<20


    """# Fig 1D
    plot_signals()
    plot_spikes()
    """
