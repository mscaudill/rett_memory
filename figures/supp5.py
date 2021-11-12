import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import rates
from scripting.rett_memory.tools import plotting, pdtools, stats

DATAPATH = paths.data.joinpath('signals_df.pkl')
ANIMALPATH = paths.data.joinpath('P80_animals.pkl')
INDEXES = [('wt',),('het',)]
#CONTEXTS = ['Neutral', 'Neutral_2']
CONTEXTS = ['Neutral_2', 'Fear', 'Fear_2']

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

if __name__ == '__main__':

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
