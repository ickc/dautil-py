import numpy as np
import operator
import pandas as pd
from functools import reduce
import scipy

# summarize ############################################################

def summarize_ndarray(data):
    '''assume data is ndarray
    return its type, dtype and shape'''
    return (type(data), data.dtype, data.shape)


def flatten_list(data):
    '''flatten nested list/tuple to arbitrary depth
    '''
    result = []
    for datum in data:
        if isinstance(datum, (list, tuple)):
            result += flatten_list(datum)
        else:
            result.append(datum)
    return result


def type_list(data):
    '''assume data is list/tuple
    may be nested list/tuple
    return the type of all elements, if they are the same
    else return None'''
    data = flatten_list(data)

    # initialize
    data_type = type(data[0])

    for datum in data:
        if not isinstance(datum, data_type):
            return None
    return data_type


def shape_list(data):
    '''assume data is list/tuple
    may be nested list/tuple
    return the shape of nested list/tuple, if regular
    else return []
    note that it will return the shape up to the inner most regular interval
    
    Example
    -------
    >>> shape_list([[[0, [1, 2]], [0, [1, 2]], [0, [1, 2]]], [[0, [1, 2]], [0, [1, 2]], [0, [1, 2]]]])
    [2, 3]
    '''
    n = [len(data)]
    m_0 = None
    for datum in data:
        if isinstance(datum, (list, tuple)):
            m = shape_list(datum)
        else:
            m = []
        if m_0 is None:
            m_0 = m
        if m_0 != m:
            return []
    return n + m_0


def summarize_list(data):
    '''assume data is list/tuple
    if types are not uniform, return 'mixed' instead
    if shape is not regular to the innermost levels, shape is ended with None
    '''
    data_type = type_list(data)
    if data_type is None:
        data_type = 'mixed'

    shape = shape_list(data)
    # detect irregular list/tuple in inner nesting levels
    # by counting no. of elements
    if len(flatten_list(data)) != reduce(operator.mul, shape, 1):
        shape.append(None)

    return (type(data), data_type, tuple(shape))


def summarize_dict(data):
    '''assume data is dict
    return a dict with values of types of values in data
    '''
    summarized = {}
    for i, j in data.items():
        if isinstance(j, dict):
            summarized[i] = summarize_dict(j)
        elif isinstance(j, np.ndarray):
            summarized[i] = summarize_ndarray(j)
        elif isinstance(j, (list, tuple)):
            summarized[i] = summarize_list(j)
        else:
            summarized[i] = type(j)
    return summarized


def summarize(data):
    '''summarize according to type
    see summarize_dict, summarize_list, summarize_ndarray
    '''
    if isinstance(data, dict):
        return summarize_dict(data)
    elif isinstance(data, (list, tuple)):
        return summarize_list(data)
    elif isinstance(data, np.ndarray):
        return summarize_ndarray(data)
    else:
        return type(data)

########################################################################

def get_map_parallel(processes):
    '''return a map function
    uses multiprocessing's Pool if processes != 1
    '''
    if processes == 1:
        return map
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=processes)
        return pool.map

########################################################################

def insert_index_level(df, level, name, value):
    '''For DataFrame df, add an index with name and value between level & (level + 1)'''
    n = df.index.nlevels
    order = list(range(n))
    order.insert(level, n)

    df = df.copy()
    df[name] = value
    df.set_index(name, append=True, inplace=True)

    return df.reorder_levels(order)

########################################################################

def df_linregress(df, grouplevel=0, regresslevel=1, regressindex=2, regressorder=2):
    '''per level ``grouplevel``, perform a linregress of each column vs. level ``regresslevel``
    assumes df has MultiIndex of 2 levels, behavior undefined otherwise.

    ``regressindex`` and ``regressorder`` control which amount the 5 outputs from scipy.stats.linregress is returned
    If ``regressindex`` is None, return all as an object, else, only return the stat at this index.
    In the latter case, the stat will be raised to the power ``regressorder``, which is useful to return $R^2$.
    TODO: if regressindex is None, return a DataFrame that the 5 elements returned at MultiIndex level-2.
    '''
    df_grouped = df.groupby(level=grouplevel)
    
    if regressindex is None:
        _linregress = lambda x: scipy.stats.linregress(x.reset_index(level=regresslevel).as_matrix())
    elif regressorder == 1:
        _linregress = lambda x: scipy.stats.linregress(x.reset_index(level=regresslevel).as_matrix())[regressindex]
    else:
        _linregress = lambda x: scipy.stats.linregress(x.reset_index(level=regresslevel).as_matrix())[regressindex]**regressorder

    dfs = (df_grouped.get_group(key).apply(_linregress).to_frame(key).transpose() for key in df_grouped.groups.keys())
    df_final = pd.concat(dfs)
    df_final.sort_index(inplace=True)
    return df_final
