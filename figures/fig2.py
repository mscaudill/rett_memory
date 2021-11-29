import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import rates
from scripting.rett_memory.tools import plotting, pdtools

DATAPATH = paths.data.joinpath('signals_df.pkl')
ANIMALPATH = paths.data.joinpath('P80_animals.pkl')
INDEXES = [('wt',),('het',)]
CONTEXTS = ['Neutral', 'Fear', 'Fear_2']
#CONTEXTS = ['Neutral', 'Neutral_2']

def rates_boxplot(groups=INDEXES, categories=CONTEXTS, cell_avg=True, 
                  ylabel='Event Rate', showfliers=False):
    """Constructs a boxplot of spike rates for Figure 2.

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

def modulations_boxplot(groups=INDEXES, categories=CONTEXTS, cell_avg=True, 
                  ylabel='Rate Reduction %', showfliers=False):
    """Constructs a boxplot of spike rates for Figure 2.

    Returns: dataframe of rates
    """

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #measure the spike rates and convert to dict for plotting
    measurer = rates.Modulation(data)
    result_df = measurer.measure(cell_avg=cell_avg)
    result_dict = pdtools.df_to_dict(result_df, groups, categories)
    #make categorical boxplot
    plotting.boxplot(result_dict, categories, groups, ylabel=ylabel,
                     showfliers=showfliers)
    return result_df

def cdf(groups=INDEXES, categories=CONTEXTS, bins=200, 
        colors=None, xlabel=None, xlims=[-1,1], **kwargs):
    """Constructs comparative cumulative distribution subplots one per
    category displaying group data.
    
    Returns: dataframe of rates
    """

    #read the dataframe and the animals to plot
    df = pd.read_pickle(DATAPATH)
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)
    #measure the spike rates and convert to a dict for plotting
    measurer = rates.Modulation(data)
    result_df = measurer.measure(cell_avg=False)
    result_dict = pdtools.df_to_dict(result_df, groups, categories)
    #make CDF plot
    plotting.cdf(result_dict, categories, groups, bins, colors=colors,
                    xlabel=xlabel, xlims=xlims, **kwargs)
    return result_df

if __name__ == '__main__':

    from scripting.rett_memory.tools import stats

    """
    #Fig 2A
    rates = rates_boxplot(cell_avg=True)
    print('WT medians')
    print(rates.loc[('wt',)].median())
    print('WT IQRS')
    print(rates.loc[('wt',)].quantile([.25, .75]))
    print('WT Counts\n', rates.loc[('wt',)].count())

    print('_____________')
    print('rtt medians')
    print(rates.loc[('het',)].median())
    print('RTT IQRS')
    print(rates.loc[('het',)].quantile([.25, .75]))
    print('RTT Counts\n', rates.loc[('het',)].count())

    print('_____________')
    print('WT STATS')
    print(stats.column_compare(rates)['wt'])
    print('_____________')
    print('RTT STATS')
    print(stats.column_compare(rates)['het'])
    """

    #Fig 2B
    mods = modulations_boxplot()
    print('WT medians')
    print(mods.loc[('wt',)].median())
    print('WT IQRS')
    print(mods.loc[('wt',)].quantile([.25, .75]))
    print('WT Counts\n', mods.loc[('wt',)].count())

    print('_____________')
    print('rtt medians')
    print(mods.loc[('het',)].median())
    print('RTT IQRS')
    print(mods.loc[('het',)].quantile([.25, .75]))
    print('RTT Counts\n', mods.loc[('het',)].count())


    print('_____________')
    print('STATS')
    print(stats.row_compare(mods))

    #results = cdf()
