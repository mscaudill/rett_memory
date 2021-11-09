import numpy as np
import pandas as pd

from scripting.rett_memory.tools.signals import crossings

def freeze_times(arr, time):
    """Returns an array of start/stop freeze times.
    
    Args:
        arr (ndarray):                      bool or binary array of freezing
                                            signal
        time (ndarray or index):            arr or pandas index of times
    """
   
    arr = np.array(arr)
    time = np.array(time)
    #get the freezing indices
    crosses = crossings(arr, level=1, interpolate=False)
    #reshape to 2-col array of starts and stops of freezes
    pairs = crosses.reshape(-1, 2)
    #remove any nan rows
    pairs = pairs[~np.isnan(pairs).any(axis=1)].astype(int)
    #compute times of pairs
    pair_times = time[pairs.flat].reshape(-1,2)
    return pair_times

def threshold(arr, time, min_time):
    """Returns an array of freezes signals where freezes of duration less
    than min_time have been removed.

    Args:
        arr (ndarray):                      bool or binary array of freezing
                                            signal
        time (ndarray or index):            arr or pandas index of times
        min_time (float):                   minimum time of freeze
    """

    arr = np.array(arr)
    time = np.array(time)
    #get the freezing times
    ftimes = freeze_times(arr, time)
    #get the durations
    durations = np.diff(ftimes, axis=1).squeeze()
    #filter the the freeze times
    filt_times = ftimes[durations < min_time] 
    #get the indices of start/stop of freezes
    idxs = np.array([np.where(time==ft)[0] for ft in filt_times.flat]).squeeze()
    paired_idxs = idxs.reshape(-1,2)
    #zero all samples of arr that lies between paired idx cols
    for pair in paired_idxs:
        arr[pair[0]:pair[1]+1]=0
    return arr

def freeze_percent(arr):
    """Returns the percentage of arr that is freezing signal."""

    data = np.array(arr)
    return np.count_nonzero(data)/len(data) * 100 if len(data) > 0 else 0

def freeze_percents(dataframe, contexts, min_time):
    """Returns a dictionary keyed on df index tuples then context with
    freezing percentages as values.
    
    Args:
        dataframe (pd.df):          behavioral dataframe (see data dir)
        contexts (seq):             prefix in df cols to compute freezing %
                                    of (e.g. Train, Neutral...) 
        min_time (float):           min time in secs for a freeze
    """

    result = dict()
    for exp in dataframe.index:
        cxt_results = dict()
        for cxt in contexts:
            signal = dataframe.loc[exp][cxt + '_freeze']
            time = dataframe.loc[exp][cxt + '_time']
            thresholded = threshold(signal, time, min_time)
            percentage = freeze_percent(thresholded)
            cxt_results[cxt] = percentage
        result[exp] = cxt_results
    return result

def filter_freezes(behavior_df, contexts, min_time, qs, groups=['geno']):
    """Returns a pandas row index of exps where the freeze percentage
    exceeds the qth quantile of freeze percentages across contexts.
    
    Args:
        behavior_df (pd.DF):        dataframe containing freeze signals
        contexts (seq):             columns of behavior_df to filter
                                    (in LH data this is Fear_2 recall)
        min_time (float):           minimum time for a freeze
        qs (seq):                   seq of percentiles of data to drop each
                                    el should be in [0,1] and have a
                                    precision of at most 2 decimal places
        groups (seq):               df levels to group df by for computing
                                    percentiles
    """

    def percentile(sub_df, q):
        """Returns a bool df where each element is compared against the qth
        percentile of the column it belongs to."""
    
        return sub_df >= sub_df.quantile(q)

    result=dict()
    #compute the freeze percents dict and convert to a dataframe
    data_dict = freeze_percents(behavior_df, contexts, min_time)
    df = pd.DataFrame(data_dict).T.sort_index()
    #create and set multindex names
    multindex = df.index.set_names(['geno', 'mouse_id', 'treatment'])
    df.set_index(multindex, inplace=True)
    for q in qs:
        #perform groupby and locate indices where all cols are true
        percentiled = df.groupby(groups).apply(percentile, q)
        indices = percentiled[np.all(percentiled, axis=1)].index
        result[round(q,2)] = indices
    return result

if __name__ == '__main__':

    from scripting.rett_memory import paths

    behavior_path = paths.data.joinpath('behavior_df.pkl')
    df = pd.read_pickle(behavior_path)
