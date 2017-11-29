from numba import jit, prange

import numpy as np
import pandas as pd


@jit(nopython=True)
def corr_custom(x, y):
    '''x, y: ndarray of dtype bool
    return: the difference between counts of x and y being agree or disagree.
    '''
    return 2. * (x == y).sum() / x.size - 1.


@jit(nopython=True)
def corr_pearson(x, y):
    '''x, y: ndarray of dtype bool
    return: pearson correlation between x, y
    Note: a.k.a. phi correlation, Matthews correlation
    Assume ``x.size == y.size``
    '''
    n = np.empty((3,))

    n[0] = x.sum()
    n[1] = y.sum()
    n[2] = (x & y).sum()

    n /= x.size

    return (n[2] - n[0] * n[1]) / np.sqrt(n[0] * (1 - n[0]) * n[1] * (1 - n[1]))


@jit(nopython=True)
def corr_kendall(x, y):
    '''x, y: ndarray of dtype bool
    return: Kendall correlation between x, y
    Assume ``x.size == y.size``
    '''
    n_01 = float((~x & y).sum())
    n_10 = float((x & ~y).sum())
    n_11 = float((x & y).sum())
    n_00 = x.size - n_01 - n_10 - n_11

    P = n_00 * n_11
    Q = n_01 * n_10
    X0 = n_00 * n_01 + n_10 * n_11
    Y0 = n_00 * n_10 + n_01 * n_11

    return (P - Q) / np.sqrt((P + Q + X0) * (P + Q + Y0))


def get_corr_matrix_func(corr_func):
    '''a higher order function that returns
    a function to calculate a correlation matrix from
    a kernel function corr_func that calculates correlations
    between 2 arrays
    '''
    @jit(nopython=True, parallel=True)
    def corr_matrix(array):
        '''array: ndarray of dtype bool
        Return correlation matrix of the columns in array
        '''
        n = array.shape[1]
        # initialize
        corr =  np.empty((n, n))
        for i in prange(n):
            corr[i, i] = 1.
            for j in range(i):
                corr[i, j] = corr_func(array[:, i], array[:, j])
                corr[j, i] = corr[i, j]
        return corr
    return corr_matrix


def df_corr_matrix(df, method='pearson'):
    '''df: DataFrame of dtypes bool
    method: custom or pearson
    return a DataFrame of the correlation matrix from columns of df
    '''
    # get corr_func
    if method in ('pearson', 'spearman', 'phi', 'matthews'):
        corr_func = corr_pearson
    elif method == 'kendall':
        corr_func = corr_kendall
    elif method == 'custom':
        corr_func == corr_custom
    else:
        import sys
        print('Unknown method {}'.format(method), file=sys.stderr)
        exit(1)

    corr_matrix_func = get_corr_matrix_func(corr_func)
    corr = corr_matrix_func(df.as_matrix())
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


@jit #(nopython=True)
def max_mask_row(array):
    result = np.zeros_like(array, np.bool_)
    for i, j in enumerate(array.argmax(axis=0)):
        result[i, j] = True
    return result


def corr_max(df, rowonly=False):
    '''df: squared correlation matrix
    rowonly: if True, only max per row, else max per either row or col
    return: a mask that is True when the correlation is max.
    '''
    # per row
    mask = max_mask_row((df - np.identity(df.shape[0])).as_matrix())
    if not rowonly:
        # or per col
        # assert mask.T == df.eq(df_max, axis=1)
        mask |= mask.T
    return pd.DataFrame(mask, index=df.index, columns=df.columns)
