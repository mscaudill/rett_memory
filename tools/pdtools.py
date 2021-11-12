import numpy as np
import pandas as pd

from collections import abc

def label_select(df, like=None, axis=1):
    """Returns row or column labels containing like substr.

    Args:
        like (str):         string to search for among row or col labels
        axis (int):         axis index to be searched (Default is columns)

    #Returns pd.Index of matches along df axis.
    """
    
    if axis not in (0,1):
        raise ValueError('Dataframe axis must be one of (0,1)')
    #if like is None return all labels on axis
    if not like:
        return df.index if axis==0 else df.columns
    #if str locate all labels containing like
    elif isinstance(like, str):
        if axis==0:
            return df.index[[like in label for label in df.index]]
        else:
            return df.columns[[like in label for label in df.columns]]
    #if seq. return like as-is
    elif isinstance(like, abc.Sequence):
        return like
    else:
        msg = '{} indexes must be "iterable" or "str" or "None" not {}'
        raise ValueError(msg.format(df.__class__.__name__,
                                        type(like)))

def rename_labels(df, new_names, inplace=True, **kwargs):
    """Renames the columns or row labels of df with new_names.

    Args:
        df (pd.Dataframe):      DataFrame whose col labels will be altered
        new_names (list):       list of new name strings
        inplace (bool):         whether to return a new dataframe obj.
        **kwargs:               keyword args passed to pd.rename
    """
    
    #create a mapper from old and new names
    zipped = zip(df.columns, new_names)
    mapper = {col: name for col, name in zip(df.columns, new_names)}
    if inplace:
        #modify inplace
        df.rename(mapper=mapper, axis=1, inplace=True)
    else:
        #return copy
        return df.rename(mapper=mapper, axis=1)

def divide(df, other, new_names):
    """Element-wise division of one dataframe by another.

    Args:
        df (pd.Dataframe):      numerator of element-wise division
        other (pd.Dataframe):   denominator of element-wise division
        new_names (list):       strings to name the cols of result

    Returns:
        resultant dataframe with columns named new_names
    """

    if df.shape != other.shape:
        msg = 'operands could not be broadcast together with shapes {} {}'
        raise ValueError(msg.format(df.shape, other.shape))

    numerator = rename_labels(df, new_names, inplace=False)
    denominator = rename_labels(other, new_names, inplace=False)
    return numerator.divide(denominator)

def df_to_dict(results_df, groups, categories):
    """Returns a dict of data keyed on group with numpy array values.
    
    Args:
        results_df (pd.DataFrame): dataframe resulting from a measurement
        groups (list):              list of tuples (geno, treatment)
        categories (list):          list of exp conditions (eg. Train, ...

    Returns: dict keyed on group with numpy array of values of shape
    num_rates X num_categories
    """
    
    #create a restult dict
    r_dict = dict()
    for exp in results_df.index:
        cxt_results = dict()
        for cxt in categories:
            data = results_df.loc[exp][cxt]
            cxt_results[cxt] = data
        r_dict[exp] = cxt_results

    results = dict()
    for group in groups:
        #get all context dicts for this geno
        dics = [cat_dic for exp, cat_dic in r_dict.items() if
                set(group).issubset(set(exp))]
        group_results = []
        #for each context gather all rates into a list
        for cxt in categories:
            group_results.append([dic[cxt] for dic in dics])
        #convert to numpy array num_animals X num_contexts
        group_results = np.array(group_results).T
        #save under this genotype key
        results[group] = group_results
    return results

def filter_df(df, mice):
    """Returns a dataframe where animals not in mice have been removed.

    Args:
        df (pd.dataframe):              dataframe to filter
        mice (list):                    list of (geno, mouse, treatment)
                                        tuples to filter df with
    """
 
    #move cell to column so we can filter by animal (geno, name, treatment)
    data = df.reset_index(level='cell')
    data = data.loc[mice]
    #add the cell column back as an index
    data = data.set_index('cell', append=True)
    return data



if __name__ == '__main__':

    df = pd.DataFrame(np.random.random((10,4)), 
                      columns='A_yes, B_no, C_yes, D_no'.split(' '),
                      index=['x_' + str(i) for i in range(10)])
    cols = str_select(df, 'yes')
    rows = str_select(df, '_4', axis=0)

    df1 = pd.DataFrame(np.random.random((10,4)), 
                      columns='A, B, C, D'.split(' '),
                      index=['x_' + str(i) for i in range(10)])
    
    div_result = divide(df, df1, new_names='Q R S T'.split(' '))
