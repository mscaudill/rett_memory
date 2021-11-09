import matplotlib.pyplot as plt
import numpy as np

def make_colors(name='tab10'):
    """Returns a list of colors from the named palette."""

    cm = plt.get_cmap(name)
    return cm.colors

def _positions(num_groups, num_categories, width, gap=None):
    """Returns arrays of x-positions for a categorical box or bar plot.

    Categorical plots are organized by categories (exp context) then by 
    animal groups (eg genotype). This function returns the x-locations to
    place boxes or bars for a comparative plot.
    """

    gap = gap if gap else 0
    #width of the box and gap
    w = width + gap
    #Below we are determing how to position each group of a category so
    #that they are centered about each index 'x-position'. This depends on
    #wheter the number of groups is even or odd
    if num_groups % 2 == 0:
        #even # of groups -> index falls at a box edge
        k = num_groups/2 -1
        left = -1 * w/2 - k * w
        right = w/2 + k * w + w 
        cat0_locs = np.arange(left, right, w)
        #FIXME added on 2-9-2020 to handle groups > categories overlap
        p = [cat0_locs + c for c in range(num_categories)]
        p = [el + 2*w*i for i, el in enumerate(p)]
        #return np.array([cat0_locs + c for c in range(num_categories)]).T
        return np.array(p).T
    else:
        #odd # of groups -> index falls at center of a box
        k = (num_groups - 1) / 2
        left = -k * w
        right = k * w + w
        cat0_locs = np.arange(left, right, w)
        return np.array([cat0_locs + c for c in range(num_categories)]).T

def bar(data, categories, groups, width=0.25, gap=0.05, colors=None,
            ylabel=None, errorbar=True, show_data=True, **kwargs):
    """Constructs a categorical barplot from a data dict.

    A categorical barplot consist of bars organized first by categories
    about 'x-position' indices then by groups about each index. This
    function plots these bars according to categroy and group order.

    Args:
        data (dict):        dict of numpy arrays keyed on
                            genotype/optogenetic condition
        categories (list):  list of string categories (e.g. Train, Fear, ..
        groups (list): list of keys of data to show on plot
        width (float):      box widths for plot (Default=0.25 axis units)
        gap (float):        gap between boxes (Default=0.05 axis units)
        colors (list):      list of colors (Default=None -> tableau palette)
        ylabel (str):       y-xais label (Default=None)
        errorbar (bool):    bool to show errorbars (Default is True)
        show_data (bool):   boot to show data points (Default is True)
        kwargs (dict):      keyword args passed to 

    Returns None, opens a matplotlib figure instance
    """

    #get the number of groups and categories
    num_groups = len(data)
    num_categories = len(categories)
    #For errorbars we will need means and standard devs
    means = {group: np.mean(arr, axis=0) for group, arr in data.items()}
    stds = {group: np.std(arr, axis=0) for group, arr in data.items()}
    #if no colors passed make tableau colors
    if colors is None:
        colors = make_colors()
    #determine the positions of the boxes
    pos = _positions(num_groups, num_categories, width, gap)
    figsize = kwargs.pop('figsize', (4,4))
    fig, ax = plt.subplots(figsize=figsize)
    for idx, group in enumerate(groups):
        #place the bars
        ax.bar(pos[idx], means[group], width=width, edgecolor='k',
                label=group, color=colors[idx], zorder=2)
        if errorbar:
            #errorbars if requested
            ecolor = kwargs.get('ecolor', 'k')
            elinewidth = kwargs.get('elinewidth', 1)
            capsize = kwargs.get('capsize', 6)
            capthick = kwargs.get('capthick', None) 
            ax.errorbar(pos[idx], means[group], stds[group], linestyle='',
                        capsize=capsize, capthick=capthick, 
                        elinewidth=elinewidth, ecolor=ecolor, 
                        zorder=3)
        if show_data:
            np.random.seed(0)
            #show data pts if requested
            n = data[group].shape[0]
            color = kwargs.get('color', 'k')
            alpha = kwargs.get('alpha', 0.4)
            s = kwargs.get('s', 24)
            edgecolor = kwargs.get('edgecolor', None)
            jits = np.random.normal(loc=0, scale=0.01, size=(n,1))
            pt_pos = np.repeat(np.expand_dims(pos[idx], axis=0), n, axis=0)
            pt_pos += jits
            plt.scatter(pt_pos, data[group], color=color,
                        edgecolor=edgecolor, alpha=alpha, s=s, zorder=3)
    #ax.grid(which='major', axis='both', color='lightgray', zorder=1)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()

    ax.legend()
    plt.show()

def boxplot(data, categories, groups, width=0.25, gap=0.05, colors=None,
        ylabel=None, scale='linear', **kwargs):
    """Constructs a categorical boxplot from a data dict.

    A categorical boxplot consist of boxes organized first by categories
    about 'x-position' indices then by groups about each index. This
    function plots these boxes according to categroy and group order.

    Args:
        data (dict):        dict of numpy arrays keyed on
                            genotype/optogenetic condition
        categories (list):  list of string categories (e.g. Train, Fear, ..
        groups (list):      list of keys of data to show on plot
        width (float):      box widths for plot (Default=0.25 axis units)
        gap (float):        gap between boxes (Default=0.05 axis units)
        colors (list):      list of colors (Default=None -> tableau palette)
        ylabel (str):       y-xais label (Default=None)
        kwargs (dict):      keyword args passed to plt.boxplot

    Returns None, opens a matplotlib figure instance
    """

    #get the number of groups and categories
    num_groups = len(data)
    num_categories = len(categories)
    #if no colors passed make tableau colors
    if colors is None:
        colors = make_colors()
    #determine the positions of the boxes
    pos = _positions(num_groups, num_categories, width, gap)
    fig, ax = plt.subplots()
    boxes = []
    for idx, group in enumerate(groups):
        boxprops=kwargs.pop('boxprops', 
                            {'facecolor': colors[idx],'linewidth': 0.5}) 
        flierprops=kwargs.pop('flierprops', 
                              {'markeredgecolor':colors[idx]})
        medianprops=kwargs.pop('medianprops', 
                               {'color':'k', 'linewidth':0.5})
        whiskerprops=kwargs.pop('whiskerprops',
                                {'linewidth':0.5})
        capprops=kwargs.pop('capprops',
                            {'linewidth':0.5})


        #handle case of nans in the data[group] array
        arr = data[group]
        group_pos = pos[idx]
        if np.isnan(arr).any():
            print('boxplot is ignoring nans')
            arrs = [x[~np.isnan(x)] for x in arr.T]
            bp = ax.boxplot(arrs, positions=pos[idx],
                                boxprops=boxprops, 
                                flierprops=flierprops, 
                                medianprops=medianprops,
                                whiskerprops=whiskerprops, 
                                capprops=capprops, 
                                widths=width, patch_artist=True, **kwargs)
            boxes.append(bp)
        else:
            #draw the group result boxes
            bp = ax.boxplot(data[group], positions=pos[idx],
                                boxprops=boxprops, 
                                flierprops=flierprops, 
                                medianprops=medianprops,
                                whiskerprops=whiskerprops, 
                                capprops=capprops, 
                                widths=width, patch_artist=True, **kwargs)
            boxes.append(bp)
    ax.set_yscale(scale)
    ax.legend([box['boxes'][0] for box in boxes], groups)
    #ax.grid(which='major', axis='both', color='lightgray', zorder=1)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(categories)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()

def probability_hist(data, categories, groups, bins, colors=None,
              xlabel=None, **kwargs):
    """Constructs a categorical probability histogram from a data dict.

    A categorical histogram consist of multiple histograms one per category
    with each histogram axis containing multiple groups of data.

    Args:
        data (dict):        dict of numpy arrays keyed on
                            genotype/optogenetic condition
        categories (list):  list of string categories (e.g. Train, Fear, ..
        groups (list):      list of keys of data to show on plot
        colors (list):      list of colors (Default=None -> tableau palette)
        xlabel (str):       x-axis label (Default=None)
        kwargs (dict):      keyword args passed to plt.hist

    Returns None, opens a matplotlib figure instance
    """

    #if no colors passed make tableau colors
    if colors is None:
        colors = make_colors()[:len(groups)]
    #Build figure and axes with sharing
    fig, axarr = plt.subplots(1, len(categories),figsize=(12.8, 4.8),
                              sharex=True, sharey=True)
    for idx, cat in enumerate(categories):
        #get a category's data across all groups
        cat_data = [data[group][:, idx] for group in groups]
        #normalize each bin by the total count to get Prob.
        weights = [np.ones(len(datum))/len(datum) for datum in cat_data]
        axarr[idx].hist(cat_data, color=colors, bins=bins, weights=weights,
                        label=groups, **kwargs)
        axarr[idx].tick_params(axis='y', which='major', labelsize=12)
        axarr[idx].tick_params(axis='x', which='major', labelsize=12)
        axarr[idx].spines['right'].set_visible(False)
        axarr[idx].spines['top'].set_visible(False)
        axarr[idx].set_title(cat, fontsize=16)
        #axarr[idx].set_xlim(0,1.0)
    axarr[0].set_ylabel('Probability', fontsize=16)
    axarr[0].set_xlabel(xlabel, fontsize=16)
    axarr[-1].legend(prop={'size':10})
    fig.tight_layout()
    plt.show()

def cdf(data, categories, groups, bins, colors=None, xlabel=None,
        xlims=[0,1], **kwargs):
    """Constructs categorical cumulative distributions from a data dict.

    A CDF consist of multiple cdfs one per category
    with each cdf axis containing multiple groups of data.

    Args:
        data (dict):        dict of numpy arrays keyed on
                            genotype/optogenetic condition
        categories (list):  list of string categories (e.g. Train, Fear, ..
        groups (list):      list of keys of data to show on plot
        colors (list):      list of colors (Default=None -> tableau palette)
        xlabel (str):       x-axis label (Default=None)
        kwargs (dict):      keyword args passed to plt.hist

    Returns None, opens a matplotlib figure instance
    """
    
    #if no colors passed make tableau colors
    if colors is None:
        colors = make_colors()[:len(groups)]
    #Build figure and axes with sharing
    fig, axarr = plt.subplots(1, len(categories),figsize=(12.8, 4.8),
                              sharex=True, sharey=True)
    for idx, cat in enumerate(categories):
        #get a category's data across all groups
        cat_data = [data[group][:, idx] for group in groups]
        axarr[idx].hist(cat_data, color=colors, bins=bins, cumulative=True,
                        density=True, histtype='step', label=groups, **kwargs)
        axarr[idx].tick_params(axis='y', which='major', labelsize=12)
        axarr[idx].tick_params(axis='x', which='major', labelsize=12)
        axarr[idx].spines['right'].set_visible(False)
        axarr[idx].spines['top'].set_visible(False)
        axarr[idx].set_title(cat, fontsize=16)
        axarr[idx].set_xlim(*xlims)
    axarr[0].set_ylabel('Probability', fontsize=16)
    axarr[0].set_xlabel(xlabel, fontsize=16)
    handles, labels = axarr[-1].get_legend_handles_labels()
    axarr[-1].legend(reversed(handles), reversed(labels), loc='lower right', 
                     prop={'size':10})
    fig.tight_layout()
    plt.show()

def grouped_cdf(data, categories, groups, bins, colors=None, xlabel=None,
                xlims=[0,1], **kwargs):
    """Constructs categorical cumulative distributions from a data dict.

    A CDF consist of multiple cdfs one per group
    with each cdf axis containing multiple categories of data.

    Args:
        data (dict):        dict of numpy arrays keyed on
                            genotype/optogenetic condition
        categories (list):  list of string categories (e.g. Train, Fear, ..
        groups (list):      list of keys of data to show on plot
        colors (list):      list of colors (Default=None -> tableau palette)
        xlabel (str):       x-axis label (Default=None)
        kwargs (dict):      keyword args passed to plt.hist

    Returns None, opens a matplotlib figure instance
    """

    #if no colors passed make tableau colors
    if colors is None:
        colors = make_colors()[:len(categories)]
    #Build figure and axes with sharing
    fig, axarr = plt.subplots(1, len(groups),figsize=(6, 3),
                              sharex=True, sharey=True)

    for idx, grp in enumerate(groups):
        #get the data across categories for this group
        cat_data = [data[grp][:,cat] for cat in range(data[grp].shape[1])]
        color = colors
        axarr[idx].hist(cat_data, color=colors, bins=bins, cumulative=True,
                        density=True, histtype='step', label=categories, 
                        **kwargs)
        axarr[idx].tick_params(axis='y', which='major', labelsize=12)
        axarr[idx].tick_params(axis='x', which='major', labelsize=12)
        axarr[idx].spines['right'].set_visible(False)
        axarr[idx].spines['top'].set_visible(False)
        axarr[idx].set_title(grp, fontsize=16)
        axarr[idx].set_xlim(*xlims)
    axarr[0].set_ylabel('Probability', fontsize=16)
    axarr[0].set_xlabel(xlabel, fontsize=16)
    handles, labels = axarr[-1].get_legend_handles_labels()
    axarr[-1].legend(reversed(handles), reversed(labels), loc='lower right', 
                     prop={'size':10})
    fig.tight_layout()
    plt.show()









