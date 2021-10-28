import sys
import pickle
import numpy as np
from pathlib import Path

from scripting.rett_memory import casia
from scripting.rett_memory.__data__ import paths

# add casia to known modules so pickles can be opened
sys.modules['casia'] = casia

from casia.segmentation.models import SVD_ICA

#GLOBALS
BASE = Path('/media/matt/Zeus/Lingjie/data/N006_wt_female_mcorrected/')

CONTEXTS = {'Train': [0, 6000], 'Fear1': [6000, 12000], 
            'Neutral1': [12000, 18000], 'Fear2': [24000, 30000],
            'Neutral2': [30000, 36000]}

SAVEDIR = paths.dataframes


def open_series():
    """Opens the series instance at BASE."""

    with open(BASE.joinpath('Series.pkl'), 'rb') as infile:
        return pickle.load(infile)

def basis_images(contexts=None, savepath=None):
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
    S.path = BASE.joinpath('N006_wt_female_mcorrected_combined.tif')
    results = dict()
    for cxt in cxts:
        print('Computing basis images for {} context'.format(cxt))
        subseries = S[slice(*CONTEXTS[cxt])]
        segmenter = SVD_ICA(subseries)
        segmenter.estimate(basis_size=220, num_tsqrs=88)
        results[cxt] = [segmenter.U, segmenter.sigma]
    path = savepath if savepath else SAVEDIR.joinpath('N006_wt_basis.pkl')
    with open(path, 'wb') as outfile:
        pickle.dump(results, outfile)
    return results

def source_images(savepath=None, basis_path=None):
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
    S.path = BASE.joinpath('N006_wt_female_mcorrected_combined.tif')
    # open the basis images and sigmas
    basis_path = basis_path if basis_path else SAVEDIR.joinpath(
                                                    'N006_wt_basis.pkl')
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
    path = savepath if savepath else SAVEDIR.joinpath('N006_wt_trainsources.pkl')
    with open(path, 'wb') as outfile:
        pickle.dump(results, outfile)
    return results

    
    

if __name__ == '__main__':

    sources = source_images(contexts=['Train'])
