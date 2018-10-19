from __future__ import print_function

from numba import jit, prange

import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal


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
        corr = np.empty((n, n))
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


@jit(nopython=True)
def partial_corr_matrix(inv):
    '''corr: inverse of a correlation matrix
    return the partial correlation matrix
    '''
    n = inv.shape[0]
    result = np.empty_like(inv)
    for i in range(n):
        for j in range(n):
            result[i, j] = -inv[i, j] / np.sqrt(inv[i, i] * inv[j, j])
    return result


def df_partial_corr_matrix(df, mode='pinv'):
    '''df: a correlation matrix
    mode: 'pinv' or 'inv', using the corresponding function in scipy.linalg
    return the partial correlation matrix
    '''
    corr = df.as_matrix()
    inv = scipy.linalg.pinv(corr) if mode == 'pinv' else scipy.linalg.inv(corr)
    corr_partial = partial_corr_matrix(inv)
    return pd.DataFrame(corr_partial, index=df.columns, columns=df.index)


def df_inv(df, mode='pinv'):
    '''df: a DataFrame where rows and columns are indentical
    mode: 'pinv' or 'inv', using the corresponding function in scipy.linalg
    return a DataFrame with the pesudo inverse matrix of the values
    '''
    inv = scipy.linalg.pinv if mode == 'pinv' else scipy.linalg.inv
    return pd.DataFrame(inv(df.as_matrix()), index=df.columns, columns=df.index)


@jit  # (nopython=True)
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


def bin_corr_relative(corr, neighbor_max):
    '''``corr`` is a correlation matrix with columns & indices sorted,
    and in regular intervals.
    Return the average of the correlations of the (k + 1)-th neighbor,
    where k ranges from 0 to neighbor_max
    '''
    interval = np.real(corr.columns[1] - corr.columns[0])
    corr_matrix = corr.as_matrix()

    result = np.empty((neighbor_max,), dtype=complex)
    # k-th neighbor
    for k in range(neighbor_max):
        neighbor = corr_matrix.diagonal(k + 1)
        result[k] = np.mean(neighbor) + 1.j * scipy.stats.sem(neighbor)
    return pd.DataFrame(result, index=np.arange(1, neighbor_max + 1) * interval, columns=['Correlation as a function of $\Delta l$'])


def get_cutoffs(data, num=50):
    '''``data``: 1d-array
    ``num``: no. of points to resolve between min and max
    of ``data``.
    use KDE to detect minimums and return an array
    of left and right cutoff
    '''
    x = np.linspace(data.min(), data.max(), num=num)
    y = scipy.stats.gaussian_kde(data)(x)

    x_mins = x[signal.argrelmin(y)]
    x_max = x[y.argmax()]
    del x, y

    cutoffs = np.empty(2)
    try:
        cutoffs[0] = x_mins[x_mins < x_max].max()
    except ValueError:
        cutoffs[0] = np.NINF
    try:
        cutoffs[1] = x_mins[x_mins > x_max].min()
    except ValueError:
        cutoffs[1] = np.inf
    return cutoffs
