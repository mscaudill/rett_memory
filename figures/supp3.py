import pickle
import numpy as np
import matplotlib.pyplot as plt

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
        ax.axis('off')
    fig.tight_layout()
    return axarr

def plot_sources(imgs=(0, 50, 100)):
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
    #for cell in cells:
    for cell in range(len(rois['boundaries'])):
        boundary = rois['boundaries'][cell]
        ax.plot(boundary[:,1], boundary[:,0], color='green')
    return ax

def plot_mipsource_rois(cells=[13, 32, 174, 145], annulus_cell=174):
    """Plots the max intensity prijection  of the source images and
    a selection of cell rois for a zoomed region.

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

def plot_cxt_mips(cxts=['Train', 'Fear', 'Neutral', 'Fear_2']):
    """Plots the source images computed from each context independently. 

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

def plot_cxt_mips_zoom(cxts=['Train', 'Fear', 'Neutral', 'Fear_2']):
    """Plots a zoomed region of the context source images along with the
    boundaries of the detected ROIs derived from the source images across
    the entire recording session.
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
            ax.plot(boundary[:,1], boundary[:,0], color='green')
    fig.tight_layout()
    return axarr





if __name__ == '__main__':

    #axarr = plot_basis()
    #axarr = plot_sources()
    #ax = plot_source_mip()
    plot_mipsource_rois()
    
    #axarr = plot_cxt_mips() 
    #axarr = plot_cxt_mips_zoom() 

    plt.show()





