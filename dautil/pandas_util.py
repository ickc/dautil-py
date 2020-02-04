import numpy as np
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
        multiindex = isinstance(index, pd.core.indexes.api.MultiIndex)
        levels = (
            [index.get_level_values(i).unique() for i in range(index.nlevels)]
            if unique else
            list(index.levels)
        ) if multiindex else (
            [index]
        )
        names = index.names if multiindex else [index.name]
        return levels, names
    
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


def df_auto_dtypes(df, debug=False, categorical=True):
    '''convert each column of the DataFrame to a smaller dtype inplace
    '''
    # I think less that a byte of different values should be
    # good candidate of being categorical data
    # n_sqrt is just to avoid the case that someone passes a
    # small DataFrame, e.g. with < 256 no. of rows
    n_cat = min(256, int(round(np.sqrt(df.shape[0]))))

    if debug:
        memory = df.memory_usage(deep=True).sum()
        print(f'Initial memory use is {memory}.')

    for name in df:
        col = df[name]

        dtype_cur = col.dtype
        if debug:
            print(f'Found column {name} with dtype {dtype_cur}.')

        # shrink int sizes
        if dtype_cur == np.int64:
            for dtype in (np.int8, np.int16, np.int32):
                if debug:
                    print(f'Try to convert it to {dtype}.')
                values = col.values
                temp = np.asarray(values, dtype=dtype)
                # replace the original column if the shrinked
                # array has no overflow
                if np.array_equal(values, temp):
                    if debug:
                        print(f'Successfully convert it to {dtype}.')
                    df[name] = temp
                    break
                elif debug:
                    print(f'Fail to convert it to {dtype}. Moving on...')
        # shrink object
        elif dtype_cur == np.object:
            # datatime
            try:
                df[name] = pd.to_datetime(col)
                if debug:
                    print('Successfully convert it to datetime.')
            except:
                if debug:
                    print('Fail to convert it to datetime. Moving on...')

            # categorical
            if categorical:
                n_unique = col.unique().size
                if n_unique <= n_cat:
                    if debug:
                        print(f"Converting it to categorical data as it has only {n_unique} unique values.")
                    df[name] = col.astype('category')
                elif debug:
                    print(f"Not converting it to categorical data as it has {n_unique} unique values.")

    if debug:
        temp = df.memory_usage(deep=True).sum()
        print(f'Final memory use is {temp}. {(memory - temp) * 100 / memory:.3g}% of original.')

    return df
