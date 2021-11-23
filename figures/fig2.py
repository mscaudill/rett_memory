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

    results = rates_boxplot(cell_avg=False)
    #results = modulations_boxplot()
    #results = cdf()
