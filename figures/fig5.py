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
    """
    #filter by top 80% of performning animals
    animals_path = paths.data.joinpath('P80_som_animals.pkl')
    with open(animals_path, 'rb') as infile:
        som_animals = pickle.load(infile)
    data = pdtools.filter_df(df, som_animals)
    """
    data = df
    areas = AreaActivity(data).measure(cell_avg=True)
    #convert to a dict for plotting and boxplot
    result_dict = pdtools.df_to_dict(areas, groups, categories)
    plotting.boxplot(result_dict, categories, groups, 
                     ylabel='Response (Area)')
    
    print(stats.row_compare(areas))
    return areas

if __name__ == '__main__':

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
    areas = som_areas()
