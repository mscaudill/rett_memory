import pandas as pd
import pickle

from scripting.rett_memory import paths
from scripting.rett_memory.metrics.behavior import filter_freezes

def build_P80_animals(df, context='Fear_2', min_time=1):
    """Saves a list of the animals for each group in the behavior dataframe
    whose freezing percentage is greater than the 20th percentile."""

    idxs = filter_freezes(df, [context], min_time, qs=[0.2])[0.2]
    path = paths.data.joinpath('P80_som_animals.pkl')
    with open(path, 'wb') as outfile:
        pickle.dump(idxs, outfile)
    return idxs


if __name__ == '__main__':

    #df = pd.read_pickle(paths.data.joinpath('behavior_df.pkl'))
    som_df = pd.read_pickle(paths.data.joinpath('som_behavior_df.pkl'))
    som_df = som_df.drop(('ssthet', 'sn554',))
    mice = build_P80_animals(som_df)
        

