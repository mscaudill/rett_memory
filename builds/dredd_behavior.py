import pandas as pd

from scripting.rett_memory import paths

def freeze_percents():
    """Creates a dataframe of freezing percentages from an excel."""

    inpath = paths.data.joinpath('dredd_behavior.xlsx')
    df = pd.read_excel(inpath, index_col=[0,1,2], 
                       names=['Neutral', 'Fear', 'Fear_2'],
                       header=0)
    df.index.set_names(['genotype', 'mouse_id', 'treatment'], inplace=True)
    outpath = paths.data.joinpath('dredd_freezes_df.pkl')
    df.to_pickle(outpath)
    return df

if __name__ == '__main__':

    df = freeze_percents()
