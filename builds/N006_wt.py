import sys
import pickle
import numpy as np
from pathlib import Path

from scripting.rett_memory import casia
from scripting.rett_memory import paths

# add casia to known modules so pickles can be opened
sys.modules['casia'] = casia

from casia.segmentation.models import SVD_ICA

#GLOBALS
BASE = Path('/media/matt/Zeus/Lingjie/data/N006_wt_female_mcorrected/')

CONTEXTS = {'Train': [0, 6000], 'Fear1': [6000, 12000], 
            'Neutral1': [12000, 18000], 'Fear2': [24000, 30000],
            'Neutral2': [30000, 36000]}

SAVEDIR = paths.data


def open_series(name='Series.pkl'):
    """Opens the series instance at BASE."""

    with open(BASE.joinpath(name), 'rb') as infile:
        #open the file and reset the path
        S = pickle.load(infile)
        S.path = BASE.joinpath('N006_wt_female_mcorrected_combined.tif')
        return S

def basis_images():
    """Computes basis images for N006_wt across all contexts."""

    S = open_series()
    segmenter = SVD_ICA(S)
    segmenter.estimate(basis_size=220, num_tsqrs=88)
    #save the basis and sigmas 
    path = SAVEDIR.joinpath('N006_wt_basis.npz')
    np.savez(path, U=segmenter.U, sigma=segmenter.sigma,
            img_shape=S.shape[1:])
    return segmenter.U, segmenter.sigma

def source_images():
    """Computes source images for N006_wt across all contexts."""

    basispath = SAVEDIR.joinpath('N006_wt_basis.npz')
    data = np.load(basispath)
    U, sigma = data['U'], data['sigma']
    S = open_series()
    segmenter = SVD_ICA(S, U=U, sigma=sigma)
    segmenter.transform(pre_drops=8, view=False)
    #save the computed sources
    path = SAVEDIR.joinpath('N006_wt_sources.npy')
    np.save(path, segmenter.sources)
    return segmenter.sources

def cxtbasis_images(contexts=None):
    """Computes the SVD basis images for each context in contexts.

    Args:
        contexts (seq):             sequence of string context names to
                                    compute SVD basis for (Default None
                                    computes basis images for all in
                                    CONTEXTS Global.
        savepath (path):            path where dict of basis images will be
                                    saved to (Default None -> save to 
                                    SAVEDIR Global under name N006_wt_basis)

    Saves: a dict of numpy arrays of basis images keyed on contexts
    """

    cxts = contexts if contexts else CONTEXTS
    S = open_series()
    # reset the path to the tif images
    #S.path = BASE.joinpath('N006_wt_female_mcorrected_combined.tif')
    results = dict()
    for cxt in cxts:
        print('Computing basis images for {} context'.format(cxt))
        subseries = S[slice(*CONTEXTS[cxt])]
        segmenter = SVD_ICA(subseries)
        segmenter.estimate(basis_size=220, num_tsqrs=88)
        results[cxt] = [segmenter.U, segmenter.sigma]
    path = SAVEDIR.joinpath('N006_wt_cxtbasis.pkl')
    with open(path, 'wb') as outfile:
        pickle.dump(results, outfile)
    return results

def cxtsource_images():
    """Computes the source images for each context from a set of basis
    images.

    Args:
        savepath (path):            path where dict of source images will be
                                    saved to (Default None -> save to 
                                    SAVEDIR Global under name
                                    N006_wt_sources)
        basis_path (path):          path to basis images dict. If None
                                    assumes name is N006_wt_basis.pkl)

    Saves: a dict of numpy arrays of source images keyed on contexts
    """

    S = open_series()
    # reset the path to the tif images
    #S.path = BASE.joinpath('N006_wt_female_mcorrected_combined.tif')
    # open the basis images and sigmas
    basis_path = SAVEDIR.joinpath('N006_wt_cxtbasis.pkl')
    with open(basis_path, 'rb') as infile:
        basis = pickle.load(infile)
    #compute the source images
    results = dict()
    for cxt, (U, sigma) in basis.items():
        print('Computing source images for {} context'.format(cxt))
        segmenter = SVD_ICA(S, U=U, sigma=sigma)
        segmenter.transform(pre_drops=3, view=False,
                            constraint_args={'level':1, 'plot':False})
        results[cxt] = segmenter.sources
    #save the sources
    path = SAVEDIR.joinpath('N006_wt_cxtsources.pkl')
    with open(path, 'wb') as outfile:
        pickle.dump(results, outfile)
    return results

def N006_wt_rois():
    """Opens the roi series for mouse N006_wt and saves the roi instances to
    a list.
    """

    S = open_series('RoiSeries.pkl')
    #get the boundaries of the rois
    boundaries = [r.centroid + r.cell_boundary for r in S.rois]
    #get the annuli of the rois
    annuli = [roi.an_coords for roi in S.rois]
    data = {'boundaries': boundaries, 'annuli': annuli}
    rois_path = SAVEDIR.joinpath('N006_wt_rois.pkl')
    with open(rois_path, 'wb') as outfile:
        pickle.dump(data, outfile)
    print('N006_wt rois written to {}'.format(rois_path))
    

if __name__ == '__main__':

    #sources = source_images()
    rois = N006_wt_rois()
