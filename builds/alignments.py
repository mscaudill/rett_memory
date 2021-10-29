import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from itertools import zip_longest
from scipy import fft
from pathlib import Path

from scripting.rett_memory.__data__ import paths
from scripting.rett_memory.dialogs import standard_dialog
from midas.readers import Tiff

class Alignment:
    """Computes & saves the motion correction shifts for a listing of
    tiff image files.

    The image files are assumed to be stored in unique dirs with each dir
    name exactly matching the image file name. The shifts are then stored as
    a pickled dict containing a numpy array of row, col shifts for the
    Train (T), Fear day 1 (F1), Neutral day 1 (N1), Fear day 2 (F2),
    & Neutral day 2 (N2) contexts. Each pickled dict is placed into the dir
    where the tiff image file is located and is named 'shifts.pkl'.
    """ 

    base = '/media/matt/Zeus/Lingjie/data'
    # extract T, F1, N1, F2, N2 from contexts
    contexts = {'T' : [0,6000], 'F1': [6000, 12000], 
                'N1': [12000, 18000], 'F2': [24000, 30000], 
                'N2': [30000, 36000]}

    def __init__(self, multiple=True):
        """Initialize with the dirs & paths to individual tiffs in base.

        Args:
            multiple (bool):        whether to open dialog to allow user to
                                    select a single dir to compute shifts 
                                    (False) or compute shifts for all tiff
                                    files in each subdirectory of base
                                    (True) (Defaut is True)
        """

        if not multiple:
            self.dirs = [standard_dialog('askdirectory',
                                         initialdir=self.base)]
        else:
            self.dirs = os.listdir(self.base)
        img_paths = []
        for d in self.dirs:
            p = Path(d)
            img_name = str(p.stem) + '_combined.tif'
            img_paths.append(p.joinpath(img_name))
        self.img_paths = img_paths

    def _open(self, path):
        """Creates a midas.Tiff instance for the tiff file at path."""

        return Tiff(path)

    def _read(self, tiff, start, stop, step=1):
        """Reads images from a tiff filepath.

        Args:
            tiff (obj):          a midas tiff instance
            start, stop, step:   start, stop and step image frames to read

        Returns: an 3-D array with images along 0th axis
        """

        ls = []
        for idx in range(start, stop, step):
            ls.append(fp.read(idx))
        return np.stack(ls)

    def _shifts(self, arr, ref_img): 
        """Computes the shift in pixels of each image in arr relative to
        a reference image using the FFT algorithm. 

        Args:
            arr (3-D array):            an array with images stored along
                                        0th axis
            ref_image (2-D array):      reference image used to correct each
                                        image in arr

        Returns: len(arr) x 2 of row, col shifts in pixels
        """
       
        fft_ref = fft.fft2(ref_img)
        fft_ref_conj = fft_ref.conjugate()
        fft_arr = fft.fft2(arr, axes=(-2, -1))
        cross_power = (fft_arr * fft_ref_conj) / (
                  abs(fft_arr) * abs(fft_ref))
        diracs = abs(np.fft.ifft2(cross_power))
        #get row, col pos of each peak for each dirac img in diracs
        locs = []
        for frame in diracs:
            locs.append(np.unravel_index(np.argmax(frame), frame.shape))
        peaks = np.array(locs)
        rows, cols = peaks[:,0], peaks[:,1]
        #shifts > 1/2 image size are negative shifts
        rows[rows > arr.shape[1] // 2] -= arr.shape[1]
        cols[cols > arr.shape[2] // 2] -= arr.shape[2]
        return peaks

    def reference(self, tiff, context):
        """Returns a mean image for all images in a given context.
        
        Args:
            tiff (obj):          a midas tiff instance
            context (str):       one of this Alignments contexts

        Returns: 2D array reference image
        """

        start, stop = self.contexts[context]
        avg = 0
        for idx in range(start, stop):
            img = tiff.read(idx)
            avg = (idx * avg + img) / (idx + 1)
        return avg

    def shifts(self, chunksize=600):
        """Opens each tiff file in each subdirectory of base & computes
        & stores frame shifts to a dict keyed on context.


        Args:
            chunksize (int):            number of frames to correct at one
                                        time (Default is 600). This number
                                        controls the amount of memory used
                                        at any given time.

        Saves: a dict of frame shifts one per subdirectory in base. Each
        dict containing context keys and a frames x 2 (row, col) array of
        pixel shifts.
        """

        for img_path in self.img_paths:
            t0 = time.perf_counter()
            print('Analyzing file: {}'.format(img_path.stem))
            results = dict()
            fp = Tiff(img_path)
            # compute reference and shifts for each context
            for context, endpts in self.contexts.items():
                starts = np.arange(endpts[0], endpts[1], chunksize)
                pts = zip_longest(starts, starts[1:], fillvalue=endpts[1])
                #compute the reference image
                ref_img = self.reference(fp, context)
                # compute the shifts
                cxt_shifts = []
                for idx, (start, stop) in enumerate(pts):
                    arr = self._read(fp, start, stop)
                    shifts = self._shifts(arr, ref_img)
                    cxt_shifts.append(shifts)
                    msg = 'Computing shifts for context: {}, {}/{} complete'
                    print(msg.format(context, idx+1, len(starts)), end='\r')
                #store to dict for this image path
                results[context] = np.concatenate(cxt_shifts, axis=0)
            fp.close_link()
            # save the results
            save_path = img_path.parent.joinpath('shifts.pkl')
            with open(save_path, 'wb') as outfile:
                pickle.dump(result, outfile)
                elapsed = time.perf_counter() - t0
            print('Shifts saved to {} in {} s'.format(save_path, elapsed))

class ShiftsDataFrame:
    """Creates a dataframe of row, col motion corrections shifts from stored
    dicts of shifts (see Alignment)."""

    base = Path('/media/matt/Zeus/Lingjie/data')

    def __init__(self):
        """Initialize ShiftsDF with dirs to locate shifts for each mouse."""

        self.dirs = [self.base.joinpath(d) for d in os.listdir(self.base)]
        self.shift_paths = [d.joinpath('shifts.pkl') for d in self.dirs]

    def _index_from_path(self, path):
        """Extracts the mouse_id and geno from the path name to use as the
        dataframes multilevel index."""

        name = path.parent.stem
        mouse_id, geno = name.split('_')[0:2]
        return [pd.Series(geno, name='geno'), 
                pd.Series(mouse_id, name='mouse_id')]

    def _open(self, path):
        """Opens the pickled dict containing shifts at path."""

        with open(path, 'rb') as infile:
            data = pickle.load(infile)
        return data

    def _dict2df(self, path):
        """Converts the dict at path into a single rowed DF."""

        data = self._open(path)
        index = self._index_from_path(path)
        s = []
        for key, arr in data.items():
            s.append(pd.Series([arr], name=key, index=index))
        return pd.concat(s, axis=1)

    def convert(self):
        """Opens and converts all the dicts in all subdirectories of base to
        a DF keyed on genotype and mouse_id with context as colums."""

        dfs = []
        for path in self.shift_paths:
            dfs.append(self._dict2df(path))
        self.df = pd.concat(dfs, axis=0)

    def save(self):
        """Pickles this Shifts DF to the __data__ dir."""
    
        if not hasattr(self, 'df'):
            self.convert()
        savepath = paths.data.joinpath('alignments.pkl')
        self.df.to_pickle(savepath)
        print('saved alignments to {}'.format(savepath))




if __name__ == '__main__':

    a = ShiftsDataFrame()
    a.convert()
    a.save()

