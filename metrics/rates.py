import numpy as np
import pandas as pd

from scripting.rett_memory.tools import pdtools, signals


class Rate:
    """A measurer of spike rates for a selection from a series dataframe.
    
    Attrs:
        df (pd.Dataframe):      dataframe rates will be computed for
        spike_cols (list):      list of spike column names
        signal_cols (list):     list of signal column names
        sample_rates (Series):  pd series of sample rates for index
        names (list):           list of resultant column names
    """

    def __init__(self, df):
        """Rate measurer initializer.

        Args:
            df (pd.Dataframe):       df of signals and spikes
        """

        self.df = df
        #get the specific columns needed
        self.signal_cols = pdtools.label_select(self.df, like='signals')
        self.spike_cols = pdtools.label_select(self.df, like='spikes')
        self.sample_rates = self.df['sample_rate']
        #set the names of the result columns
        self.names = [c.replace('_spikes','') for c in self.spike_cols]

    def measure(self, cell_avg=False):
        """Returns a dataframe of firing rates."""

        #get signal length
        sig_lens = self.df[self.signal_cols].applymap(signals.nanlen)
        #get the signal durations
        sig_durations = sig_lens.divide(self.sample_rates, axis=0)
        #get the number of spikes
        n_spikes = self.df[self.spike_cols].applymap(len)
        #compute the rates
        rates = pdtools.divide(n_spikes, sig_durations, self.names)
        if cell_avg:
            #groupby geno and mouse id only (drop treatment, cell id)
            group_idx = rates.index.names[:-2]
            mean_rates = rates.groupby(by=group_idx).apply(np.mean)
            return mean_rates
        
        return rates


class Modulation:
    """A measurer of the spike rate modulation for a selection from a
    dataframe containing spikes.

    Attrs:
        df (pd.Dataframe):      dataframe rates will be computed for
        spike_cols (list):      list of spike column names
        names (list):           list of resultant column names
        reference (str):        spike column of dataframe to use as
                                reference

    """

    def __init__(self, df, reference='Neutral'):
        """Modulation measurer initializer.

        Args:
            df (pd.Dataframe):          dataframe of signals and spikes
            reference (str):            column name of df to use as
                                        modulation reference data

        """

        self.df = df
        #get specfic columns needed
        self.spike_cols = pdtools.label_select(self.df, like='spikes')
        #get the sub dataframe
        self.df = self.df[self.spike_cols]
        #set the names of the results columns
        self.names = [c.replace('_spikes','') for c in self.spike_cols]
        #set reference col
        self.reference_name = reference + '_spikes'

    def measure(self, cell_avg=False):
        """Returns a dataframe of modulations.
        
        modulation for a context and cell is:
        (num_spikes - num_ref_spikes) / (num_spikes + num_ref_spikes)
        """
        
        #get the number of spikes for all spike list in df
        nspikes_df = self.df.applymap(len)
        #small number to ensure non-zero denominators
        epsi = 1e-10
        #compute numerator and denominator
        numerator = nspikes_df.subtract(nspikes_df[self.reference_name],
                axis=0)
        denominator = nspikes_df.add(nspikes_df[self.reference_name] + epsi,
                axis=0)
        #divide
        result = pdtools.divide(numerator, denominator, self.names)

        if cell_avg:
            group_idx = result.index.names[:-1]
            mean_modulations = result.groupby(by=group_idx).apply(np.mean)
            return mean_modulations
        return result
