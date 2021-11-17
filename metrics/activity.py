import numpy as np
import pandas as pd
import scipy as sp

from operator import itemgetter
from scipy.signal import windows, find_peaks
from functools import partial
from copy import copy

from scripting.rett_memory.tools import pdtools


def scipy_window(win_func, **kwargs):
    """Constructs a scipy signal window.
    
     Args:
            win_func (scipy window):        func in scipy's window module
            **kwargs (dict):                keyword args necessary for
                                            specific win_func

    Notes: see scipy.signal.windows for list of available windows and
    their keyword arguments.
    """

    try:
        window = getattr(windows, win_func.__name__)(**kwargs)
    except AttributeError:
        #show available windows
        availables = [name for (name, obj) in getmembers(windows) if
                      isfunction(obj)]
        msg = ("module '{}' has no attribute '{}'; available windows",
               "are:\n {}")
        raise AttributeError(msg.format(windows, win_kind, availables))
    return window

def gauss_window(M, std, norm=True):
    """Constructs a scipy Gaussian window that may be normalized.

    M (int):            number of pts in window
    std (float):        width of window
    norm (bool):        normalize the window to unit area
    """

    if norm:
        amplitude = 1 / np.sqrt(2 * np.pi * std**2)
    else:
        amplitude = 1
    return amplitude * windows.gaussian(M, std)

def build_trains(spikes_df, signals_df):
    """Returns a dataframe of full spike trains.

    Args:
        spikes_df:          a dataframe of spike index arrays
        signals_df:         a dataframe of signals

    This function converts spike index arrays into arrays of signal len
    with 1's at spike indexes and 0's elsewhere. It will also have NaNs at
    the same positions as in signals.
    """

    def indices_to_train(spike_ls, signal):
        """Converts spike indices list into train array 1 x len(signal)."""
        #shallow copy to avoid changes to dataframes
        ls = spike_ls.copy()
        train = signal.copy()
        #initialize train from signal
        train[~np.isnan(train)] = 0
        #place spikes
        train[ls] = 1
        return train
    
    #list of panda series results
    series_ls = []
    for sp_col, sig_col in zip(spikes_df.columns, signals_df.columns):
        #get the context name
        cxt = sp_col.split('_spikes')[0]
        #convert this column of spikes and signals to lists
        spikes = spikes_df[sp_col].to_list()
        signals = signals_df[sig_col].to_list()
        #compute trains
        trains = [indices_to_train(spike_ls, signal) for spike_ls, signal 
                    in zip(spikes, signals)]
        #append to results
        series_ls.append(pd.Series(trains, index=spikes_df.index, name=cxt))
    #conacatenate to form a dataframe of spike trains
    return pd.concat(series_ls, axis=1)

def convolve(train, window):
    """Convolves a spike train with a window."""

    return np.convolve(train, window, mode='same')

class CellActivity:
    """Measures the activity of each spike index list in the series
    dataframe by convolution with a scipy window.

    Attrs:
        series_df (pd.DataFrame):       series dataframe
    """

    def __init__(self, series_df):
        """Initialize this measurement."""

        self.df = series_df
        #get column labels for spikes and signals
        spike_cols = pdtools.label_select(self.df, like='spikes')
        signal_cols = pdtools.label_select(self.df, like='signals')
        #extract the spikes and signals sub dataframe
        self.spikes_df = self.df[spike_cols]
        self.signals_df = self.df[signal_cols]

    def measure(self, window):
        """Returns the convolved activities for each spike list in the
        series dataframe.
        
        Args:
            window (sequence):         m-point sequence of window values
        """

        sample_rate = np.unique(self.df.sample_rate)[0]
        #build trains
        trains = build_trains(self.spikes_df, self.signals_df)
        #convolve trains with window
        activities = trains.applymap(partial(convolve, window=window))
        #convolved rate is in spikes/sample so multiply by rate
        return sample_rate*activities, trains

class AreaActivity:
    """Measures the activity of the the signals in the series dataframe
    using the area as a surrogate for activity.

    Attrs:
        series_df (pd.DataFrame):   series dataframe
    """

    def __init__(self, series_df):
        """Initialize this measurement."""

        self.df = series_df
        #get column labels for signals
        signal_cols = pdtools.label_select(self.df, like='signals')
        #extract the signals sub dataframe
        self.signals_df = self.df[signal_cols]
        #set the names of the results columns
        self.names = [c.replace('_signals','') for c in signal_cols]

    def measure(self, cell_avg=False, **kwargs):
        """Returns the summed trapezoidal area of the positive signal
        deflections.

        kwargs passed to numpy trapz
        """

        fs = np.unique(self.df.sample_rate)[0]
        def area(signal):
            """Returns the positive area 1-D array"""

            s = signal[~np.isnan(signal)]
            s[s<0] = 0
            x = np.linspace(0, len(s)/fs, len(s))
            return np.trapz(s, x=x, **kwargs) 
        #compute the areas and rename the cols
        areas = self.signals_df.applymap(area)
        pdtools.rename_labels(areas, self.names)
        if cell_avg:
            group_idx = areas.index.names[:-1]
            mean_areas = areas.groupby(by=group_idx).apply(np.mean)
            return mean_areas
        return areas

class NetworkActivity:
    """Measures the mean activity of all spike index lists in a
    dataframe by convolution with a scipy window.

    Attrs:
        series_df (pd.DataFrame):       series dataframe
        std (float):                    width in samples of gaussian 
                                        window used to compute cell
                                        activities
    """

    def __init__(self, series_df, std=5):
        """Initialize this activity measurement."""

        self.df = series_df
        #get column labels for spikes and signals
        spike_cols = pdtools.label_select(self.df, like='spikes')
        signal_cols = pdtools.label_select(self.df, like='signals')
        #extract the spikes and signals sub dataframe
        self.spikes_df = self.df[spike_cols]
        self.signals_df = self.df[signal_cols]
        #build cell activities
        npoints = 10 * std
        window = gauss_window(M=npoints, std=std)
        self.activities, _ = CellActivity(series_df).measure(window)

    def measure(self):
        """Returns the average activity of spike index lists in series
        dataframe.
        
        Returns a new dataframe of mean activities indexed by a tuple (geno,
        mouse).
        """

        def combine(group):
            """Helper that takes a sub dataframe 'group' and returns the
            mean across the rows."""

            means = []
            for col in group.columns:
                means.append([np.mean(group[col], axis=0)])
            return pd.DataFrame(dict(zip(group.columns, means)))

        #do a group-apply-combine operations
        group_index = self.activities.index.names[:-1] #ignore cell 
        net_activities = self.activities.groupby(by=group_index).apply(combine)
        #drop the 0th pooled index
        net_activities = net_activities.droplevel(level=len(group_index))
        return net_activities


class NetworkBursts:
    """Estimates the peak samples of network bursts.

    Attrs:
        df (pd.DataFrame):      pandas dataframe containing spike indices
        spike_cols (list):      list of spike column names
        signal_cols (list):     list of signal column names
        sample_rates (Series):  pd series of sample rates for index
        names (list):           list of resultant column names
    """

    def __init__(self, df):
        """Initialize this location metric with a dataframe of series data
        and a behavioral filter.
        """

        self.df = df
        #get the specific columns needed
        self.signal_cols = pdtools.label_select(self.df, like='signals')
        self.spike_cols = pdtools.label_select(self.df, like='spikes')
        #get the sub dataframes and Series' needed
        self.sample_rate = np.unique(self.df['sample_rate'])[0]
        #set the names of the result columns
        self.names = [c.replace('_spikes','') for c in self.spike_cols]

    def activity(self, width):
        """Returns the cell and network activities by gaussian convolution.

        Args:
            width (int):        width of gaussian convolving window in
                                samples
        """

        self.std = width
        metric = NetworkActivity(self.df, width)
        cell_activities = metric.activities
        network_activities = metric.measure()
        return cell_activities, network_activities

    def threshold(self, activities, shifts, repeats, nstds):
        """Determines the minimum height of a significant network burst.
        
        The threshold is determined by constructing repeat number of
        surrogate datasets where each activity trace has been randomly
        shifted and the network activity recomputed. Events exceeding nstds
        of the maximum network amplitude observed in the surrogate data are
        considered significant.

        Args:
            activities (df):       dataframe of cell act traces
            shifts (tuple):        min/max amt to shift each cell act trace
            repeats (int):         number of times to repeat act trace 
                                   shifting
            nstds (float):         number of stds from mean 
        """
        
        #remove the cell level to be averaged over
        acts = activities.droplevel('cell')
        
        def athreshold(exp, cxt):
            """Returns a single threshold for a given exp and context of the
            cell activities."""

            data = np.stack(acts.loc[exp][cxt], axis=0)
            maxes = []
            for rep in range(repeats):
                np.random.seed(rep) #make shifting reproducible
                rolls = zip(data, np.random.randint(*shifts, len(data)))
                #compute cell surrogates and network surrogate
                surrogates = np.array([np.roll(*tup) for tup in rolls]) 
                net_surrogates = np.nanmean(surrogates, axis=0)
                maxes.append(np.nanmax(net_surrogates))
            #return the threshold as nstds above the mean
            return np.mean(maxes) + nstds * np.std(maxes)

        #call athreshold on each row, col index
        dresults = dict()
        for exp in acts.index.unique():
            dresults[exp] = {cxt: athreshold(exp, cxt) for cxt in acts}
        #convert to a df and return
        results = pd.DataFrame.from_dict(dresults, orient='index')
        results.index.names = acts.index.names
        return results

    def peaks(self, data, height):
        """Locates peaks in data containing NaNs.
        
        Args:
            data (np.arr)       array possibly containing nans 
            height:             passed to scipy find_peaks
        """

        arr = data[~np.isnan(data)]
        #make a transformer
        nans = np.where(np.isnan(data))[0]
        indices = np.arange(len(data), dtype=float)
        indices[nans] = np.NAN
        transform = indices[~np.isnan(indices)]
        #compute the peaks of the nan_free arr
        peaks = find_peaks(arr, height)[0]
        #transform the peaks into the nan_present indices
        return (transform[peaks]).astype(int)

    def measure(self, gauss_width=5, shift_range=[500,1500], repeats=10,
                nstds=1.5):
        """Locates the peaks of the network activity.
        
        Args:
            gauss_width (float):    width of convolving Gaussian
            shift (2-seq):          two-el seq of min and max shift amounts
                                    used to construct surrogate data
            repeats (int):          number of surrogate datas to construct
            nstds (float):          number of stds above mean for peak to be
                                    significant (default is 1.5)

        This method finds the statistically significant peaks by finding the
        maximum peaks that occur in randomized surrogate data constructed by
        shifting each cell activity trace relative to the others by some 
        random integer of sample in shift_range.
        """
        
        #compute the cell and network activities dataframes
        cell_acts, network_acts = self.activity(gauss_width)
        self.cell_activities, self.net_activities = cell_acts, network_acts
        #compute the thresholds dataframe
        thresholds = self.threshold(cell_acts, shift_range,
                                         repeats, nstds)
        #for each exp and context compute the peaks of the net activity
        dresults = dict()
        for exp in thresholds.index:
            cxt_results = dict()
            for cxt in thresholds.columns:
                #get net activity
                data = network_acts.loc[exp][cxt]
                #get the threshold for this exp and context
                threshold = thresholds.loc[exp][cxt]
                #find peaks
                cxt_results[cxt] = self.peaks(data, threshold)
            dresults[exp] = cxt_results
        #convert to df return
        results = pd.DataFrame.from_dict(dresults, orient='index')
        results.index.names = network_acts.index.names
        return results


class Coactivity:
    """A metric that measures the number of cells participating in the
    largest network burst.

    Attrs:
        df (pd.DataFrame):          dataframe containing spike indices
        spike_cols (list):      list of spike column names
        signal_cols (list):     list of signal column names
        sample_rate (int):      sample rate assumed consistent across df
        names (list):           list of resultant column names
    """

    def __init__(self, df):
        """Initialize with a dataframe containing spike indices."""

        self.df = df
        #get the specific columns needed
        self.signal_cols = pdtools.label_select(self.df, like='signals')
        self.spike_cols = pdtools.label_select(self.df, like='spikes')
        #get the sub dataframes and Series' needed
        self.sample_rate = np.unique(self.df['sample_rate'])[0]
        #set the names of the result columns
        self.names = [c.replace('_spikes','') for c in self.spike_cols]

    def activity(self, width):
        """Returns the cell and network activities by gaussian convolution.

        Args:
            width (int):        width of gaussian convolving window in
                                samples
        """

        self.std = width
        metric = NetworkActivity(self.df, width)
        cell_activities = metric.activities
        network_activities = metric.measure()
        return cell_activities, network_activities

    def threshold(self, activities, shifts, repeats, nstds):
        """Determines the minimum height of a significant network burst.
        
        The threshold is determined by constructing repeat number of
        surrogate datasets where each activity trace has been randomly
        shifted and the network activity recomputed. Events exceeding nstds
        of the maximum network amplitude observed in the surrogate data are
        considered significant.

        Args:
            activities (df):       dataframe of cell act traces
            shifts (tuple):        min/max amt to shift each cell act trace
            repeats (int):         number of times to repeat act trace 
                                   shifting
            nstds (float):         number of stds from mean 
        """
        
        #remove the cell level to be averaged over
        acts = activities.droplevel('cell')
        
        def athreshold(exp, cxt):
            """Returns a single threshold for a given exp and context of the
            cell activities."""

            data = np.stack(acts.loc[exp][cxt], axis=0)
            maxes = []
            for rep in range(repeats):
                np.random.seed(rep) #make shifting reproducible
                rolls = zip(data, np.random.randint(*shifts, len(data)))
                #compute cell surrogates and network surrogate
                surrogates = np.array([np.roll(*tup) for tup in rolls]) 
                net_surrogates = np.nanmean(surrogates, axis=0)
                maxes.append(np.nanmax(net_surrogates))
            #return the threshold as nstds above the mean
            return np.mean(maxes) + nstds * np.std(maxes)

        #call athreshold on each row, col index
        dresults = dict()
        for exp in acts.index.unique():
            dresults[exp] = {cxt: athreshold(exp, cxt) for cxt in acts}
        #convert to a df and return
        results = pd.DataFrame.from_dict(dresults, orient='index')
        results.index.names = acts.index.names
        return results

    def largest_peak(self, network_activities, thresholds):
        """Returns a dataframe of the largest network peak.
        
        Args:
            network_activities (df):        dataframe of network activites
            thresholds (df):                dataframe of height thresholds
                                            significant network act 
                                            deflections
        """

        def largest(data, height):
            """Returns index of the largest peak in data exceeding height."""

            data[np.isnan(data)] = 0
            x, props = find_peaks(data, height)
            heights = props['peak_heights']
            if not x.any():
                return np.NAN
            else:
                return sorted(zip(x, heights), key=itemgetter(1))[-1][0]

        #find largest peak for each exp and context
        dresults = dict()
        for exp in network_activities.index:
            cxt_results = dict()
            for cxt in network_activities.columns:
                data = network_activities.loc[exp][cxt]
                threshold = thresholds.loc[exp][cxt]
                cxt_results[cxt] = largest(data, threshold)
            dresults[exp] = cxt_results
        #convert to a df and return
        results = pd.DataFrame.from_dict(dresults, orient='index')
        results.index.names = network_activities.index.names
        return results

    def measure(self, band=50, level=1, width=5, shifts=[500, 1500], 
                repeats=10, nstds=1.5):
        """Returns a dataframe of network densities.
        
        Args:
            band (int):         number of samples centered on peak to look
                                for coactive cells (Default=50 => cells with
                                activity upto 25 samples prior or after
                                peak are included in measure)
            level (float):      firing rate in Hz for cell to be considered
                                as part of network burst (Default is 1 Hz)
            width (int):        width of convolving gaussian for cell
                                activity 
            shifts (seq):       min and max shift amounts for constructing
                                surrogate data for determining significant 
                                peaks (Default is between 500 and 1500
                                samples)
            repeats (int):      number of surrogate datasets to construct to
                                determine network peak significance (Default
                                is 10 repeats)
            nstds (float):      number of stds above the mean for a peak to
                                be significant (Default is 1.5 standard 
                                deviations)
        """

        #compute activities and thresholds
        cell_acts, net_acts = self.activity(width=width)
        thresholds = self.threshold(cell_acts, shifts, repeats, nstds)
        peaks = self.largest_peak(net_acts, thresholds)
        #create a band array
        band = np.array([-band//2, band//2])
        #determine which cells are active in the band
        dresults = dict()
        for exp in peaks.index:
            cxt_results = dict()
            for cxt in peaks.columns:
                if np.isnan(peaks.loc[exp][cxt]):
                    cxt_results[cxt] = np.NAN
                    continue
                #get all activities and slice out band
                acts_arr = np.stack(cell_acts.loc[exp][cxt], axis=0)
                sl = slice(*(band + peaks.loc[exp][cxt]).astype(int))
                max_vals = np.max(acts_arr[:,sl], axis=1)
                #count those above level
                density = np.count_nonzero(max_vals > level) / len(acts_arr)
                cxt_results[cxt] = density
            dresults[exp] = cxt_results
        #convert to a df and return
        results = pd.DataFrame.from_dict(dresults, orient='index')
        results.index.names = net_acts.index.names
        return results


class CoactiveCells(Coactivity):
    """Returns a boolean dataframe indicating whether individual cells are
    part of the coactive ensemble."""

    def measure(self, band=20, level=1, width=5, shifts=[500, 1500], 
                repeats=10, nstds=1.5):
        """Measures whether each cell exceeds or fails to exceed a threshold
        for being considered a coactive cell.

        Args:
            band (int):         number of samples centered on peak to look
                                for coactive cells (Default=50 => cells with
                                activity upto 25 samples prior or after
                                peak are included in measure)
            level (float):      firing rate in Hz for cell to be considered
                                as part of network burst (Default is 1 Hz)
            width (int):        width of convolving gaussian for cell
                                activity 
            shifts (seq):       min and max shift amounts for constructing
                                surrogate data for determining significant 
                                peaks (Default is between 500 and 1500
                                samples)
            repeats (int):      number of surrogate datasets to construct to
                                determine network peak significance (Default
                                is 10 repeats)
            nstds (float):      number of stds above the mean for a peak to
                                be significant (Default is 1.5 standard 
                                deviations)
        """

        #compute activities and thresholds
        cell_acts, net_acts = self.activity(width=width)
        thresholds = self.threshold(cell_acts, shifts, repeats, nstds)
        peaks = self.largest_peak(net_acts, thresholds)
        #create a band array
        band = np.array([-band//2, band//2])
        #determine which cells are active in the band
        for exp in cell_acts.index:
            for cxt in cell_acts.columns:
                sl = slice(*(band + peaks.loc[exp[:-1]][cxt]).astype(int))
                arr = np.array(cell_acts.loc[exp][cxt])[sl]
                if arr.size > 0:
                    #FIXME should I compare with level?
                    if np.max(arr) >= thresholds.loc[exp[:-1]][cxt]:
                        cell_acts.loc[exp][cxt] = True
                    else:
                        cell_acts.loc[exp][cxt] = False
                else:
                    cell_acts.loc[exp][cxt] = False
        return cell_acts


if __name__ == '__main__':

    from scripting.rett_memory import paths
    import pickle

    ANIMALPATH = paths.data.joinpath('P80_animals.pkl')


    df = pd.read_pickle(paths.data.joinpath('signals_df.pkl'))
    with open(ANIMALPATH, 'rb') as infile:
        animals = pickle.load(infile)
    data = pdtools.filter_df(df, animals)

    metric = CoactiveCells(data)
    cc = metric.measure()
