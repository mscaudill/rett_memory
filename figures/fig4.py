import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, binomtest
from scripting.rett_memory.tools.stats import mwu
from scripting.rett_memory import paths

PYR_IPSC_FREQ_PATH = paths.data.joinpath('pc_sipscs_freqs.pkl')
PYR_IPSC_AMP_PATH = paths.data.joinpath('pc_sipscs_amps.pkl')
SOM_EPSC_FREQ_PATH = paths.data.joinpath('som_sepsc_freqs.pkl')
SOM_EPSC_AMP_PATH = paths.data.joinpath('som_sepsc_amps.pkl')

def barplot(data, names=None, ylabel=None, **kwargs):
    """Barplots each named sequence in data dictionary with std error bars.

    Args:
        data: dict
                a dict containing 1-D array like values to barplot
        
        names: sequence of strings
                Keys of data to barplot. If None plot all the groups in data
        
        ylabel: str
                optional string label to apply to y-axis

    Returns: matplotlib axis instance
    """ 

    names = list(data.keys()) if not names else names
    fig, ax = plt.subplots()

    for idx, name in enumerate(names):

        # fetch the values omitting any nans
        values = np.array(data[name])
        values = values[~np.isnan(values)]
    
        # compute error bars and barplot
        yerr = np.std(values)
        ax.bar(idx, np.mean(values), yerr=yerr, capsize=6, edgecolor='k')
        
        # add scatter of values to plot
        x = idx * np.ones(len(values))
        ax.scatter(x, values, zorder=2, color='k', alpha=0.5, **kwargs)
    
    # set labels and return axis
    ax.set_ylabel(ylabel)
    plt.xticks(np.arange(len(names)), names)
    return ax


def pyr_ipscs():
    """Returns a barplot axis for Figure 4C of the paper comparing sIPSCs in
    pyramidal cells of WT and RTT mice."""

    with open(PYR_IPSC_FREQ_PATH, 'rb') as infile:
        pyr_ipsc_freq = pickle.load(infile)

    with open(PYR_IPSC_AMP_PATH, 'rb') as infile:
        pyr_ipsc_amp = pickle.load(infile)

    
    ax1 = barplot(pyr_ipsc_freq, ylabel='sIPSC Frequency (Hz)')
    ax2 = barplot(pyr_ipsc_amp, ylabel='sIPSC Amplitude (pA)')

    print('sIPSC FREQUENCY DATA')
    for group, ls in pyr_ipsc_freq.items():
        name = group
        mean = np.nanmean(np.array(ls))
        se = np.nanstd(np.array(ls))
        n = np.count_nonzero(~np.isnan(ls))
        print('Group {} mean = {} || std = {} || n = {}'.format(
              name, mean, se, n))

    print('Kruskal Omnibus Test')
    print(kruskal(*pyr_ipsc_freq.values()), end='\n\n')

    print('sIPSC AMPLITUDE DATA')
    for group, ls in pyr_ipsc_amp.items():
        name = group
        mean = np.nanmean(np.array(ls))
        se = np.nanstd(np.array(ls))
        n = np.count_nonzero(~np.isnan(ls))
        print('Group {} mean = {} || std = {} || n = {}'.format(
              name, mean, se, n))

    print('Kruskal Omnibus Test')
    print(kruskal(*pyr_ipsc_amp.values()), end='\n\n')

    plt.show()


def som_epscs():
    """ Returns a barplot axis for Figure 4C of the paper comparing sEPSCs in
    som cells of WT and RTT mice."""

    with open(SOM_EPSC_FREQ_PATH, 'rb') as infile:
        som_epsc_freq = pickle.load(infile)

    with open(SOM_EPSC_AMP_PATH, 'rb') as infile:
        som_epsc_amp = pickle.load(infile)

    
    ax1 = barplot(som_epsc_freq, ylabel='sEPSC Frequency (Hz)')
    ax2 = barplot(som_epsc_amp, ylabel='sEPSC Amplitude (pA)')

    print('sEPSC FREQUENCY DATA')
    for group, ls in som_epsc_freq.items():
        name = group
        mean = np.nanmean(np.array(ls))
        se = np.nanstd(np.array(ls))
        n = np.count_nonzero(~np.isnan(ls))
        print('Group {} mean = {} || std = {} || n = {}'.format(
              name, mean, se, n))

    print('Mann Whitney U Tests')
    print('WT SOM vs. RTT SOM MeCP2-')
    print(mwu(*[som_epsc_freq[x] for x in ['som', 'som_rett_neg']]))
    print('RTT SOM MeCP2+ vs. RTT SOM MeCP2-')
    print(mwu(*[som_epsc_freq[x] for x in ['som_rett_pos', 'som_rett_neg']]))

    print('-'*50)
    print('sEPSC AMPLITUDE DATA')
    for group, ls in som_epsc_amp.items():
        name = group
        mean = np.nanmean(np.array(ls))
        se = np.nanstd(np.array(ls))
        n = np.count_nonzero(~np.isnan(ls))
        print('Group {} mean = {} || std = {} || n = {}'.format(
              name, mean, se, n))

    print('Mann Whitney U Test')
    print('WT SOM vs. RTT SOM MeCP2-')
    print(mwu(*[som_epsc_amp[x] for x in ['som', 'som_rett_neg']]))
    print('RTT SOM MeCP2+ vs. RTT SOM MeCP2-')
    print(mwu(*[som_epsc_amp[x] for x in ['som_rett_pos', 'som_rett_neg']]))

    plt.show()


def connectivity_percentages():
    """Compares the connectivity percentages for Figure 4 J and L of the
    paper."""

    #PYR to OLM (Figure 4J)
    expected = 10/81
    measured = 2
    trials = 68
    pyr_olm_result = binomtest(measured, n=trials, p=expected)

    #OLM to PYR (Figure 4L)
    expected = 8/81
    measured = 9
    trials = 68
    olm_pyr_result = binomtest(measured, n=trials, p=expected)
    return pyr_olm_result, olm_pyr_result


if __name__ == '__main__':

    #pyr_ipscs()
    #som_epscs()

    pyr_olm_result, olm_pyr_result = connectivity_percentages()
