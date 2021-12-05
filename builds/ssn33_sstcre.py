import sys
import pickle
import numpy as np
from pathlib import Path

from scripting.rett_memory import casia
from scripting.rett_memory import paths

# add casia to known modules so pickles can be opened
sys.modules['casia'] = casia

from casia.segmentation.models import SVD_ICA

BASE = Path('/media/matt/Zeus/Lingjie/data/ssn33_sstcre/')

SAVEDIR = paths.data


def open_series(name='Series.pkl'):
    """Opens the series instance at BASE."""

    with open(BASE.joinpath(name), 'rb') as infile:
        #open the file and reset the path
        S = pickle.load(infile)
        S.path = BASE.joinpath(
                'ssn33_sstcre_fear_SST+_mcorrected_combined.tif')
        return S

def basis_images():
    """Computes basis images for ssn33_sstcre across all contexts."""

    S = open_series()
    segmenter = SVD_ICA(S)
    segmenter.estimate(basis_size=100, num_tsqrs=88)
    #save the basis and sigmas 
    path = SAVEDIR.joinpath('ssn33_sstcre_basis.npz')
    np.savez(path, U=segmenter.U, sigma=segmenter.sigma,
            img_shape=S.shape[1:])
    return segmenter.U, segmenter.sigma


def source_images():
    """Computes source images for ssn33_sstcre across all contexts."""

    basispath = SAVEDIR.joinpath('ssn33_sstcre_basis.npz')
    data = np.load(basispath)
    U, sigma = data['U'], data['sigma']
    S = open_series()
    segmenter = SVD_ICA(S, U=U, sigma=sigma)
    segmenter.transform(pre_drops=13, view=False)
    #save the computed sources
    path = SAVEDIR.joinpath('ssn33_sstcre_sources.npy')
    np.save(path, segmenter.sources)
    return segmenter.sources

if __name__ == '__main__':

    #u, sigma = basis_images()
    sources = source_images()
