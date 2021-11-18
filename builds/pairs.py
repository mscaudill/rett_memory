import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.widgets import Slider

from scripting.rett_memory.metrics import activity
from scripting.rett_memory.dialogs import standard_dialog

class PearsonPairs:
    """A builder of pairs of cells with a high Pearson correlation coeff.

    Attrs:
        df (pd.DataFrame):              dataframe containing spike data
        window (scipy window):          a window for transforming a spike
                                        train into an activity trace by 
                                        convolution (Default is Gauss)
        threshold (float):              correlation threshold for Pears
                                        Coeff in [0,1] (Default is 0.3)

    This metric also includes a method for viewing correlated pairs
    """

    def __init__(self, df, window=None, threshold=0.3):
        """Initialize this builder."""

        self.df = df

        if not window:
            self.window = activity.gauss_window(M=50, std=5, norm=True)
        else:
            self.window = window
        self.threshold = threshold

    def build(self):
        """Returns a dataframe of correlated cells pairs.

        The returned dataframe has the same indexing as the series_df except
        one less level since cells are dropped.
        """

        self.activities, _ = activity.CellActivity(self.df).measure(self.window)
        pairs = defaultdict(dict)
        for exp in np.unique(self.df.index.droplevel('cell')):
            for cxt in self.activities.columns:
                #convert acts of this exp and context to np arr
                sub_df = self.activities.loc[exp][cxt]
                data = np.stack(sub_df.to_numpy(), axis=0)
                #zero nans and find pairwise correlations
                data[np.isnan(data)] = 0
                corrs = np.corrcoef(data)
                #need only consider upper half to find > threshold
                corrs[np.tril_indices(corrs.shape[0], k=1)] = 0
                locs = np.where(corrs > self.threshold)
                #stack cell_i, cell_j corrs along cols
                idxs = np.stack(locs, axis=1)
                #get the actual correlation and bundle with locs
                c_vals = corrs[locs].reshape(len(idxs), -1)
                result = np.concatenate([idxs, c_vals], axis=1)
                pairs[exp][cxt] = result
        self.pairs_df = pd.DataFrame.from_dict(pairs, orient='index',
                                      columns=self.activities.columns)
        self.pairs_df.index.names = self.activities.index.names[:-1]
        return self.pairs_df

    def view(self, exp, cxt):
        """Opens an interactive matplotlib pyplot for viewing the pairs for
        a specfified exp and context of this metrics dataframe. """

        #create a figure and fetch activity traces and correlated pairs
        fig, ax = plt.subplots(figsize=(10,6))
        data = np.stack(self.activities.loc[exp][cxt].to_numpy(), axis=0)
        pair_arr = self.pairs_df.loc[exp][cxt]
        fig.suptitle('Geno: {}, Mouse: {}, Context: {}'.format(*exp, cxt))

        #helper to update our figure on slider selection
        def update(value):
            #get slider value and clear any previous draws
            value = int(value)
            ax.clear()
            #get the pair and the correlation of the pair
            cell, other, corr = pair_arr[value]
            cell, other = int(cell), int(other)
            #get the traces and draw (invert one to make vis. easier)
            trace1, trace2 = data[cell], data[other]
            line1, = ax.plot(trace1)
            line2, = ax.plot(-1*trace2)
            #annotate with correlation value and add cell ids to legend
            ax.annotate('Correlation: {:.2f}'.format(corr), (.82, 1.02), 
                        xycoords='axes fraction')
            ax.legend((line1, line2), ('Cell {}'.format(cell), 
                                       'Cell {}'.format(other)),
                                        loc='upper right')
            plt.draw()

        #first draw is 0th pair
        update(0)
        #create a slider and set callback
        slider_ax = plt.axes([0.15, 0.02, .73, .03], facecolor='white')
        slider = Slider(slider_ax, label='Pair', valmin=0, 
                        valmax=len(pair_arr)-1, valinit=0,  valstep=1, 
                        valfmt='%1.0f', facecolor='#1f77b4')
        slider.on_changed(update)
        plt.show()

    def save(self, path=None):
        """Saves the Pearson pairs dataframe to path."""

        path = path if path else standard_dialog('asksaveasfilename')
        print('Saving pairs to {}'.format(path))
        self.pairs_df.to_pickle(path)

if __name__ == '__main__':

    from scripting.rett_memory import paths
    from scripting.rett_memory.tools import pdtools

    df = pd.read_pickle(paths.data.joinpath('signals_df.pkl'))
    with open(paths.data.joinpath('P80_animals.pkl'), 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)

    builder = PearsonPairs(data)
    pairs = builder.build()
    #builder.save(paths.data.joinpath('correlated_pairs_df.pkl'))

