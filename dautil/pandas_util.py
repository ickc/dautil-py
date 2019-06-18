import pandas as pd


def get_slice(df, name, no_levels=False, no_indexslice=False):
    '''Given ``name`` of an index from the MultiIndex
    from ``df``, return the ``level`` this name is at,
    ``levels`` of all levels except this name is at,
    and an ``indexslice`` that slices all MultiIndex

    Note: ``indexslice`` is of type list. Useful to be
    reassigned at ``level`` to another slice.
    When passed to pandas.DataFrame, need to be casted to
    tuple first.
    '''
    level = df.index.names.index(name)
    n = df.index.nlevels

    if not no_levels:
        # get a list of all levels but level (of 'name')
        levels = list(range(n))
        levels.pop(level)
    else:
        levels = None

    if not no_indexslice:
        # create empty slice of length nlevels
        indexslice = [slice(None)] * n
    else:
        indexslice = None

    return level, levels, indexslice


def df_to_ndarray(df, unique=False):
    ''' convert DataFrame with MultiIndex to ndarray

    `unique`: if True, take only the unique values from the MultiIndex levels

    columns and index can be either MultiIndex of Index (all combinations allowed)

    assume the MultiIndex is a product, else error occurs when reshape

    return values, levels, names

    essentially values should equals to df.to_xarray().to_array().values
    but faster (and less safe)
    '''
    assert df.index.is_monotonic
    assert df.columns.is_monotonic

    def get_index_levels_names(index):
        multiindex = isinstance(index, pd.core.index.MultiIndex)
        levels = (
            [index.get_level_values(i).unique() for i in range(index.nlevels)]
            if unique else
            list(index.levels)
        ) if multiindex else (
            [index]
        )
        names = index.names if multiindex else [index.name]
        return levels, names

    columns = df.columns
    
    col_levels, col_names = get_index_levels_names(df.columns)
    row_levels, row_names = get_index_levels_names(df.index)
    levels = col_levels + row_levels
    names = col_names + row_names

    ns = [level.size for level in levels]

    # recall that DataFrame is column major
    values = df.values.T.reshape(ns)

    return values, levels, names


def ndarray_to_series(values, levels, names):
    '''convert ndarray to Series

    almost inverse of df_to_ndarray but return a Series instead.
    unstack can be used to freely move some of the index to columns
    to recover the original df.

    i.e. ndarray_to_series(*df_to_ndarray(df)) ~ df.stack()
    where all the levels in columns are stacked in the beginning of the MultiIndex levels
    '''
    s = pd.Series(values.flatten(), index=pd.MultiIndex.from_product(levels))
    s.index.names = names
    return s
