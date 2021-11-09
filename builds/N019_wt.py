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
BASE = Path('/media/matt/Zeus/Lingjie/data/N019_wt_female_mcorrected/')

CONTEXTS = {'Train': [0, 6000], 'Fear1': [6000, 12000], 
            'Neutral1': [12000, 18000], 'Fear2': [24000, 30000],
            'Neutral2': [30000, 36000]}

SAVEDIR = paths.data


def open_series(name='Series.pkl'):
    """Opens the series instance at BASE."""

    with open(BASE.joinpath(name), 'rb') as infile:
        #open the file and reset the path
        S = pickle.load(infile)
        S.path = BASE.joinpath('N019_wt_female_mcorrected_combined.tif')
        return S

def basis_images():
    """Computes basis images for N019_wt across all contexts."""

    S = open_series()
    segmenter = SVD_ICA(S)
    segmenter.estimate(basis_size=220, num_tsqrs=88)
    #save the basis and sigmas 
    path = SAVEDIR.joinpath('N019_wt_basis.npz')
    np.savez(path, U=segmenter.U, sigma=segmenter.sigma,
            img_shape=S.shape[1:])
    return segmenter.U, segmenter.sigma

def source_images():
    """Computes source images for N019_wt across all contexts."""

    basispath = SAVEDIR.joinpath('N019_wt_basis.npz')
    data = np.load(basispath)
    U, sigma = data['U'], data['sigma']
    S = open_series()
    segmenter = SVD_ICA(S, U=U, sigma=sigma)
    segmenter.transform(pre_drops=10, view=False)
    #save the computed sources
    path = SAVEDIR.joinpath('N019_wt_sources.npy')
    np.save(path, segmenter.sources)
    return segmenter.sources


if __name__ == '__main__':

   #U, sigma = basis_images() 
   sources = source_images()
