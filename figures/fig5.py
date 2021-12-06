import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scripting.rett_memory import paths
from scripting.rett_memory.metrics import behavior
from scripting.rett_memory.tools import plotting, pdtools, stats
from scripting.rett_memory.metrics.activity import AreaActivity

def som_MIP():
    """Plots the max intensity projection of all the source images for
    animal ssn33_sstcre mouse for Figure 5A."""

    path = paths.data.joinpath('ssn33_sstcre_sources.npy')
    data = np.load(path)
    mip = np.max(data, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mip, cmap='gray')
    plt.show()

def plot_signals():
    """Plots the standardized signal of a subset of cells for animal
    ssn33_sstcre for Figure 5B."""

    df = pd.read_pickle(paths.data.joinpath('som_signals_df.pkl'))
    signals = df.loc[('sstcre', 'ssn33',)]['Fear_2_signals']
    signals = np.array([row for row in signals])[np.arange(7), :]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4))
    time = np.linspace(0, 300, 6000)
    for idx, signal in enumerate(signals):
        ax0.plot(time, signal + 5 * idx)
    #add a scale bars
    ax0.plot([40,40], [-1, 0], color='k') 
    ax0.plot([40, 45], [-1, -1], color='k')
    ax0.set_xlim([35, 305])
    ax1.plot(time, signals[0])
    ax1.plot([40,40], [-1, 0], color='k') 
    ax1.plot([40, 45], [-1, -1], color='k')
    ax1.set_xlim([35,100])
    plt.show()

def som_areas():
    """Boxplots the area below the standardized fluorescence curves for WT
    and RTT Som cells."""

    categories = ['Neutral','Fear','Fear_2']
    groups = [('sstcre',), ('ssthet',)]
    df = pd.read_pickle(paths.data.joinpath('som_signals_df.pkl'))
    areas = AreaActivity(df).measure(cell_avg=True)
    #convert to a dict for plotting and boxplot
    result_dict = pdtools.df_to_dict(areas, groups, categories)
    plotting.boxplot(result_dict, categories, groups, 
                     ylabel='Response (Area)', showfliers=False)
    return areas

def dredd_rescue():
    """Boxplots the freezing percentages of mice treated with a dredd or
    vehicle for Figure 5E of the paper."""

    path = paths.data.joinpath('dredd_freezes_df.pkl')
    df = pd.read_pickle(path)
    df.sort_index()
    #groups = np.unique(df.index.droplevel('mouse_id'))
    groups = [('sst-cre', 'mcherry'),
              ('sst-cre', 'hm4d'),
              ('sst-cre_rtt', 'mcherry'),
              ('sst-cre_rtt', 'hm3d')]
    data = pdtools.df_to_dict(df, groups=groups, categories=df.columns)
    plotting.boxplot(data, categories=df.columns, groups=groups,
                     ylabel='Freezing (%)', showfliers=False)
    return df, data

if __name__ == '__main__':

    from scipy.stats import kruskal

    plt.ion()
    # Figure 5A
    """
    som_MIP()
    """

    #Figure 5B
    """
    plot_signals()
    """

    #Figure 5C
    """
    df = som_areas()
    print('SOM')
    print('medians')
    print(df.loc[('sstcre')].median())
    print('IQRS')
    print(df.loc[('sstcre')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('sstcre')].count())
    print('_____________\n')
    
    print('RTT SOM')
    print('medians')
    print(df.loc[('ssthet')].median())
    print('IQRS')
    print(df.loc[('ssthet')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('ssthet')].count())
    print('_____________\n')

    print('Statistics')
    print(stats.row_compare(df))
    """

    #Figure 5E
    df, data_dict = dredd_rescue()

    print('########  som-cre mcherry  ########')
    print('medians')
    print(df.loc[('sst-cre',slice(None), 'mcherry')].median())
    print('IQRS')
    print(df.loc[('sst-cre',slice(None), 'mcherry')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('sst-cre', slice(None), 'mcherry')].count())
    print('_____________\n')

    print('########  som-cre HM4D  ########')
    print('medians')
    print(df.loc[('sst-cre',slice(None), 'hm4d')].median())
    print('IQRS')
    print(df.loc[('sst-cre',slice(None), 'hm4d')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('sst-cre', slice(None), 'hm4d')].count())
    print('_____________\n')

    print('########  som-cre RTT mcherry  ########')
    print('medians')
    print(df.loc[('sst-cre_rtt',slice(None), 'mcherry')].median())
    print('IQRS')
    print(df.loc[('sst-cre_rtt',slice(None), 'mcherry')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('sst-cre_rtt', slice(None), 'mcherry')].count())
    print('_____________\n')

    print('########  som-cre RTT HM3D  ########')
    print('medians')
    print(df.loc[('sst-cre_rtt',slice(None), 'hm3d')].median())
    print('IQRS')
    print(df.loc[('sst-cre_rtt',slice(None), 'hm3d')].quantile([.25, .75]))
    print('Counts\n', 
          df.loc[('sst-cre_rtt', slice(None), 'hm3d')].count())
    print('_____________\n')

    print('Statistics')
    s = stats.row_compare(df)
    for context, vals in s.items():
        print('------{}-----'.format(context))
        print(vals)
        print('\n')

    print('Neutral Omnibus')
    arrs = [x[:,0] for x in data_dict.values()]
    kresult = kruskal(*arrs)
    print(kresult)
