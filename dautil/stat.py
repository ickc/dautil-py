from numba import jit, prange

import numpy as np
import pandas as pd


@jit(nopython=True)
def corr_biserial(x, y):
    '''x, y: ndarray of dtype bool
    return: biserial correlation between x, y
    '''
    return 2. * np.count_nonzero(x == y) / x.shape[0] - 1.


@jit(nopython=True)
def corr_pearson(x, y):
    '''x, y: ndarray of dtype bool
    return: pearson correlation between x, y
    '''
    n = np.empty((2, 2), dtype=np.int32)

    # strictly speaking, n[0, 0] and N should be calculated by the following lines
    # n[0, 0] = (~x & ~y).sum()
    # N = n.sum()
    # but n[0, 0] is not used anywhere below.
    # so just for simplicity, n[0, 0] is used to store N instead
    n[0, 0] = x.shape[0]

    n[0, 1] = (~x & y).sum()
    n[1, 0] = (x & ~y).sum()
    n[1, 1] = (x & y).sum()

    mean = np.empty((2,))
    mean[0] = n[1, :].sum()
    mean[1] = n[:, 1].sum()
    mean /= n[0, 0]

    return (n[1, 1] / n[0, 0] - np.prod(mean)) / np.sqrt(np.prod(mean - np.square(mean)))


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
            for j in np.arange(i):
                corr[i, j] = corr_func(array[:, i], array[:, j])
                corr[j, i] = corr[i, j]
        return corr
    return corr_matrix


def df_corr_matrix(df, method='biserial'):
    '''df: DataFrame of dtypes bool
    method: biserial or pearson
    return a DataFrame of the correlation matrix from columns of df
    '''
    corr_func = corr_biserial if method == 'biserial' else corr_pearson
    corr_matrix_func = get_corr_matrix_func(corr_func)
    corr = corr_matrix_func(df.as_matrix())
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def corr_max(df, rowonly=False):
    '''df: squared correlation matrix
    rowonly: if True, only max per row, else max per either row or col
    return: a mask that is True when the correlation is max.
    '''
    df_max = (df - np.identity(df.shape[0])).max()

    # per row
    mask = df.eq(df_max, axis=0)
    if not rowonly:
        # or per col
        mask |= df.eq(df_max, axis=1)
    return mask
