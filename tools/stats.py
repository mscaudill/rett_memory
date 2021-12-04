import itertools
import warnings
import numpy as np
import pandas as pd

from collections import namedtuple
from scipy import stats

def Stat(test_name, **attrs):
    """Creates a statistical result named tuple instance with attrs."""
        
    Test = namedtuple(test_name, sorted(attrs))
    return Test(**attrs)
    
def non_nan(x):
    """Returns array where rows with an NaN value have been removed."""

    bool_arr = np.isnan(x).any(axis=1)
    return np.count_nonzero(bool_arr), x[~bool_arr]

def column_compare(df, **kwargs):
    """Computes for each index in dataframe a Wilcoxon signed rank test for
    each pair of columns in dataframe.
    
    Args:
        kwargs (dict):      keyword args for scipy's wilcoxon test

    Returns: dict of Stat named tuples keyed on index with attrs:
        T: the test's statistic
        p: the test's pvalue
        n: the test's sample length
        effect: the test's effect size
    """

    #drop the mouse_id level and build combination of columns
    ds = ['mouse_id', 'cell'] if 'cell' in df.index.names else ['mouse_id']
    #new dataframe with levels dropped and construct col combos
    df = df.droplevel(ds)

    def compare(df, index,  **kwargs):
        """Returns a dict of Wilcoxon signed rank test across column pairs
        in df at a single row index."""
        
        sub_df = df.loc[index]
        comparisons = dict()
        for pair in itertools.combinations(df.columns, 2):
            #get the test inputs
            samples = np.array(sub_df[list(pair)])
            #remove sample comparisons with nans
            nan_cnt, samples = non_nan(samples)
            if nan_cnt:
                msg='Dropping {} NaN sample comparisons for: Index {}, pair {}'
                warnings.warn(msg.format(nan_cnt, index, pair))
            n = samples.shape[0]
            x, y = list(samples.T)   
            #compute test and get effect size
            t, p = stats.wilcoxon(x, y, **kwargs)
            rank_sum = np.sum(range(n))
            #effect = (rank_sum - t) / rank_sum - t / rank_sum
            effect = t / rank_sum
            #make a stat and save
            stat = Stat('wilcoxon', T=t, p=p, n=n, effect=effect)
            comparisons.update({pair: stat})
        return comparisons
    
    indices = np.unique(df.index)
    return {idx: compare(df, idx, **kwargs) for idx in indices}

def row_compare(df, columns=None, **kwargs):
    """ Computes for each column in dataframe a Mann-Whitney U test for
    each pair of groups in dataframe.
    
    Args:
        kwargs (dict):      keyword args for scipy's mann-whitney test

    Returns: dict of Stat named tuples keyed on index with attrs:
        U: the test's statistic
        p: the test's pvalue
        n1: the test's sample one length
        n1: the test's sample two length
        effect: the test's effect size
"""

    #drop the mouse_id level and build combination of columns
    ds = ['mouse_id', 'cell'] if 'cell' in df.index.names else ['mouse_id']
    #new dataframe with levels dropped and construct col combos
    df = df.droplevel(ds)

    def compare(df, column, **kwargs):
        """Returns a dict of Mann-Whitney U test for a single column of the
        dataframe."""

        lens = [len(df.loc[idx]) for idx in np.unique(df.index)]
        if (np.array(lens) < 20).any():
            msg = 'Sample size too small for normal approximation. '\
                    'Consult table for p-value'
            warnings.warn(msg)
        
        comparisons = dict()
        for pair in itertools.combinations(np.unique(df.index), 2):

            x, y = df.loc[pair[0]][column], df.loc[pair[1]][column]
            x_nan, y_nan = np.isnan(x), np.isnan(y)
            if x_nan.any() or y_nan.any():
                msg='Dropping {} samples for pair {} and column {}'
                cnt = np.count_nonzero(np.concatenate((x_nan, y_nan)))
                warnings.warn(msg.format(cnt, pair, column))
            x, y = x[~x_nan], y[~y_nan]
            n1, n2 = len(x), len(y)
            u, p = stats.mannwhitneyu(x, y, **kwargs)
            u = min(u, len(x)*len(y) - u)
            effect = u / (n1*n2)
            stat = Stat('WMW', U=u, p=p, n1=n1, n2=n2, effect=effect)
            comparisons.update({pair: stat})
        return comparisons

    cols = df.columns if not columns else columns
    return {col: compare(df, col, **kwargs) for col in cols}


def mwu(arr1, arr2, **kwargs):
        """Computes the mann-whitney U statistic and asymptotic p-value

        Args:
            arr1, arr2:             1-D arrays to compare
            kwargs:                 passed to scipy stats.mannwhitneyu
        """

        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        #ignore NAN policy
        x = arr1[~np.isnan(arr1)]
        y = arr2[~np.isnan(arr2)]
        n, m = len(x), len(y)
        u, p = stats.mannwhitneyu(arr1, arr2, **kwargs)
        return min(u, n * m - u), p

