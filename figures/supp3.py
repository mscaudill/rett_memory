import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

from scripting.rett_memory import paths

"""
Sub figures for supplemental figure 3 describing imaging methods.
"""

def plot_basis(imgs=(20, 50, 100)):
    """Plots a sample of basis images from mouse N006_wt_female.

    The basis images are basis for the entire recording session (i.e. across
    all environmental contexts.)
    """
 
    data = np.load(paths.data.joinpath('N006_wt_basis.npz'))
    fig, axarr = plt.subplots(1, len(imgs))
    for ax, im_num in zip(axarr, imgs):
        shape = data['img_shape']
        #reshape the basis vectors into images (imgs x height x width)
        basis = data['U'].swapaxes(0,1).reshape(-1, *shape, order='F')
        img = basis[im_num]
        ax.imshow(img, cmap='gray')
        #ax.axis('off')
    fig.tight_layout()
    return axarr

def plot_sources(imgs=(0, 25, 50, 70, 100)):
    """Plots a sample of independent component images from mouse
    N006_wt_female.

    The source images are computed from the entire recording session (i.e.
    across all environmental contexts.
    """

    data = np.load(paths.data.joinpath('N006_wt_sources.npy'))
    fig, axarr = plt.subplots(1, len(imgs))
    for ax, source_num in zip(axarr, imgs):
        ax.imshow(data[source_num], cmap='gray')
        ax.axis('off')
    fig.tight_layout()
    return axarr

def plot_source_mip():
    """Plots the max intensity projection of the source images from mouse
    N006_wt_female and all the detected rois.

    The source images are computed from the entire recording session (i.e.
    across all environmental contexts.
    """

    data = np.load(paths.data.joinpath('N006_wt_sources.npy'))
    mip = np.max(data, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mip, cmap='gray')
    with open(paths.data.joinpath('N006_wt_rois.pkl'), 'rb') as infile:
        rois = pickle.load(infile)
    #plot the cell boundaries and annuli
    """
    for cell in range(len(rois['boundaries'])):
        boundary = rois['boundaries'][cell]
        ax.plot(boundary[:,1], boundary[:,0], color='green')
    """
    return ax

def plot_mipsource_rois(cells=[13, 32, 174, 145], annulus_cell=174):
    """Plots the max intensity prijection  of the source images and
    a selection of cell rois for a zoomed region for mouse N006_wt.

    The source images are computed from the entire recording session (i.e.
    across all environmental contexts.
    """

    data = np.load(paths.data.joinpath('N006_wt_sources.npy'))
    mip = np.max(data, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mip, cmap='gray')
    ax.set_xlim(260, 360)
    ax.set_ylim(270, 350)
    ax.invert_yaxis()
    with open(paths.data.joinpath('N006_wt_rois.pkl'), 'rb') as infile:
        rois = pickle.load(infile)
    #plot the cell boundaries and annuli
    for cell in range(len(rois['boundaries'])):
        boundary = rois['boundaries'][cell]
        annulus = rois['annuli'][cell]
        if cell == annulus_cell:
            ax.plot(boundary[:,1], boundary[:,0], color='orange',
                    linewidth=2)
            ax.plot(annulus[:,1], annulus[:,0], linestyle=' ', marker='s',
                color='b')
        else:
            ax.plot(boundary[:,1], boundary[:,0], color='green', linewidth=2)
    return ax

def plot_cxt_mips(cxts=['Fear', 'Neutral', 'Fear_2']):
    """Plots the source images computed from each context independently for
    mouse N006_wt. 

    The source images are computed indepently for each context.
    """

    path = paths.data.joinpath('N006_wt_cxtsources.pkl')
    with open(path, 'rb') as infile:
        data = pickle.load(infile)
    mips = {name: np.max(sources, axis=0) for name, sources in data.items()}
    fig, axarr = plt.subplots(1, len(cxts), sharex=True, sharey=True)
    for cxt, ax in zip(cxts, axarr):
        ax.imshow(mips[cxt], cmap='gray')
    fig.tight_layout()
    return axarr

def plot_cxt_mips_zoom(cxts=['Fear', 'Neutral', 'Fear_2']):
    """Plots a zoomed region of the context source images along with the
    boundaries of the detected ROIs derived from the source images across
    the entire recording session for mouse N006_wt
    """

    path = paths.data.joinpath('N006_wt_cxtsources.pkl')
    with open(path, 'rb') as infile:
        data = pickle.load(infile)
    mips = {name: np.max(sources, axis=0) for name, sources in data.items()}
    fig, axarr = plt.subplots(1, len(cxts), sharex=True, sharey=True)
    # load the roi boundaries and annuli
    with open(paths.data.joinpath('N006_wt_rois.pkl'), 'rb') as infile:
        rois = pickle.load(infile)
    # plot the zoomed context sources
    for cxt, ax in zip(cxts, axarr):
        ax.imshow(mips[cxt], cmap='gray')
        ax.set_xlim(325, 425)
        ax.set_ylim(385, 485)
        ax.invert_yaxis()
        # plot the roi boundaries
        for cell in range(len(rois['boundaries'])):
            boundary = rois['boundaries'][cell]
            ax.plot(boundary[:,1], boundary[:,0], color='green', linewidth=2)
    fig.tight_layout()
    return axarr

def plot_displacements_N006(cxts=['T', 'F1', 'N1', 'F2']):
    """Plots a time-series of concatenated displacements one for each
    context in cxts for mouse N006_wt."""

    # open the dataframe of alignments for all mice
    path = paths.data.joinpath('alignments.pkl')
    with open(path, 'rb') as infile:
        df = pickle.load(infile)
    #mulitply by um/pix and compute displacements
    df = df * 1.2
    data = df.loc[('wt', 'N006')].to_dict()
    displacements = [np.sqrt(np.sum(data[cxt]**2, 1)) for cxt in cxts]
    displacements = np.concatenate(displacements)
    #plot displacements
    fig, ax = plt.subplots()
    time = np.linspace(0, len(displacements)/20, len(displacements))/60
    ax.plot(time, displacements, color='tab:gray')
    ax.set_xlabel('Time (mins)')
    ax.set_ylabel('Displacement (um)')
    [ax.axvline(5*i, color='k', linestyle=':', zorder=-1) 
            for i in range(1, len(cxts))]
    ax.set_ylim(0, None)
    return ax

def hist_N006_displacements(cxts=['F1', 'N1', 'F2']):
    """Constructs a histogram of displacements one per context in cxts for
    mouse N006_wt."""

    # open the dataframe of alignments for all mice
    path = paths.data.joinpath('alignments.pkl')
    with open(path, 'rb') as infile:
        df = pickle.load(infile)
    #mulitply by um/pix and compute displacements
    df = df * 1.2
    data = df.loc[('wt', 'N006')].to_dict()
    displacements = [np.sqrt(np.sum(data[cxt]**2, 1)) for cxt in cxts]
    displacements = np.concatenate(displacements)
    print(np.count_nonzero(displacements < 5)/ len(displacements))
    #plot displacements
    fig, ax = plt.subplots()
    ax.hist(displacements, bins=6, color='tab:gray', align='left',
            rwidth=.95)
    ax.set_xlabel('Displacement (um)')
    ax.set_ylabel('Num. Frames')
    return ax

def hist_displacements(cxts=['T', 'F1', 'N1', 'F2']):
    """Constructs a histogram of all displacements across all mice for all
    contexts in cxts."""

    # open the dataframe of alignments for all mice
    path = paths.data.joinpath('alignments.pkl')
    with open(path, 'rb') as infile:
        df = pickle.load(infile)
    #mulitply by um/pix and compute displacements
    df = df * 1.2
    results = []
    for index in df.index:
        data = df.loc[index]
        displacements = [np.sqrt(np.sum(data[cxt]**2, 1)) for cxt in cxts]
        displacements = np.concatenate(displacements)
        results.append(displacements)
    results = np.stack(results, axis=0)
    fig, axarr = plt.subplots(1, len(cxts), figsize=(10,2), sharey=False)
    fig2, axarr2 = plt.subplots(1, len(cxts), figsize=(10,2), sharey=True)
    for idx, cxt in enumerate(cxts):
        subarr = results[:, idx*6000:(idx+1)*6000].flatten()
        print(cxt, np.count_nonzero(subarr < 10)/ len(subarr))
        axarr[idx].hist(subarr, color='tab:gray', align='left', rwidth=0.95)
        axarr2[idx].hist(subarr, color='tab:gray', bins=100, linewidth=2, 
                histtype='step', cumulative=True, density=True)
    return axarr, results


if __name__ == '__main__':

    plt.ion()
    #axarr = plot_basis()
    #axarr = plot_sources()
    #ax = plot_source_mip()
    #ax = plot_mipsource_rois()
    #axarr = plot_cxt_mips() 
    #axarr = plot_cxt_mips_zoom() 
    #ax = plot_displacements_N006()
    #ax = hist_N006_displacements()
    ax, results = hist_displacements(cxts=['T','F1', 'N1', 'F2'])
    plt.show()





