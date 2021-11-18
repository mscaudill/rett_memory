import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from scripting.rett_memory import casia
from scripting.rett_memory import paths

# add casia to known modules so pickles can be opened
sys.modules['casia'] = casia

FP = Path('/home/matt/python/nri/data/rett_recall/wt_rett_df.pkl')

def build_rois(path=FP):
    """Returns a dataframe of Roi centroids, cell bouhdaries and annulus
    boundaries from a casia processed dataframe."""

    df = pd.read_pickle(path)
    df = df['rois']
    index = df.index

    def extract(obj):
        """Extracts the centroid and boundaries for each Roi instance stored
        on df."""

        return obj.centroid, obj.cell_boundary, obj.an_coords

    df = df.apply(extract)
    df = pd.DataFrame(df.to_list(), columns=['centroid', 'cell_boundary',
                                             'annulus_boundary'])
    df = df.set_index(index)
    return df

if __name__ == '__main__':

    df = build_rois()
    df.to_pickle(paths.data.joinpath('rois_df.pkl'))
