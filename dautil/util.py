from numba import jit
import numpy as np
import operator
import pandas as pd
from functools import reduce
import scipy
import types

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
    try:
        data_type = type(data[0])
    except IndexError:
        return None

    for datum in data[1:]:
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
    if not isinstance(data, (list, tuple)):
        return []

    # initialized
    try:
        m = shape_list(data[0])
    except IndexError:
        return []

    n = [len(data)]
    
    for datum in data[1:]:
        if m != shape_list(datum):
            return n

    return n + m


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


def get_variables(module):
    '''Return a list of variable names from ``module``.
    Currently, callables and modules are ignored.
    '''
    return [item for item in dir(module) if not
            (item.startswith("__") or
             isinstance(getattr(module, item), types.ModuleType) or
             callable(getattr(module, item)))]


def assert_dict(input1, input2, rtol=1.5e-09, atol=1.5e-09, verbose=False):
    '''
    recursively assert into a dictionary
    if the value is a numpy array, assert_allclose, else assert equal.
    '''
    for key in input1:
        if isinstance(input1[key], dict):
            if verbose:
                print('asserting {}'.format(key))
            assert_rec(input1[key], input2[key], rtol=rtol, atol=atol, verbose=verbose)
        elif isinstance(input1[key], np.ndarray):
            if verbose:
                print('asserting {}'.format(key))
            np.testing.assert_allclose(input1[key], input2[key], rtol=rtol, atol=atol)
        else:
            if verbose:
                print('asserting {}'.format(key))
            assert input1[key] == input2[key]

# numpy array ##########################################################


@jit(nopython=True, nogil=True)
def arange_inv(array):
    '''array: assumed to be an output of numpy.arange
    return (start, stop, step) which used to create this array
    '''
    start = array[0]
    step = (array[-1] - array[0]) / (array.size - 1)
    stop = array[-1] + step
    return start, stop, step


@jit(nopython=True, nogil=True)
def linspace_inv(array):
    '''array: assumed to be an output of numpy.linspace
    return (start, stop, num) which used to create this array
    '''
    return array[0], array[-1], array.size


def get_box(array):
    '''array: numpy.ndarray
    return: per dim. in array, find the min. and max. index s.t.
    the other dim. are identically zero.

    This might be used when one want to trim out the empty "boundary" of an array to reduce size.
    In principle, a trimmed array with this attribute can reconstruct the original array.

    Note:

    - This might not make sense if ndim = 1
    - This is relatively slow, that uses numpy only to walk through the array.
    '''
    ndim = array.ndim

    result = np.empty((ndim, 2), dtype=np.int64)

    for i in range(ndim):
        axis = list(range(ndim))
        axis.pop(i)
        axis = tuple(axis)
        # all but the i-axis
        minmax = ~np.all(array == 0, axis=axis)
        minmax_index = minmax * np.arange(minmax.shape[0])
        minmax_index = minmax_index[minmax_index != 0]
        for j in (0, -1):
            result[i, j] = minmax_index[j]

    return result


def get_outer_box(x, y):
    '''Given boxes from the output of get_box,
    return the smallest box that contains both.
    '''
    return np.column_stack((np.minimum(x, y)[:, 0], np.maximum(x, y)[:, 1]))


def to_levels(array, mask=None, dtype=np.uint8, ranges=(0, 255), fix_origin=False):
    '''mapping the values of ``array`` between its min. and max. to ``levels``
    with dtype ``dtype``
    '''
    if mask is not None:
        array = array[mask]

    _min = 0. if fix_origin else array.min()
    scale = (ranges[1] - ranges[0]) / ((array.max() - _min) or 1.)

    result = (array * scale + (ranges[0] - _min * scale)).astype(dtype)

    if mask is not None:
        result2 = np.zeros_like(mask, dtype=dtype)
        result2[mask] = result
        return result2
    else:
        return result


def unpackbits(data, flags):
    '''
    data: 1d-array of type int
    flags: 1d-array of "bits", e.g. numpy.array([1, 2, 4, 8, ...])
    Return
    ------
    2d-array, where each element is the unpacked array of bits per datum.
    Note
    ----
    With flags = 2**numpy.arange(7, -1, -1), and data has dype = uint8,
    this is the same as numpy.unpackbits
    '''
    return (data[:, None] & flags) != 0


@jit(nopython=True, nogil=True)
def running_mean(x, n):
    '''
    return: array of the moving average of x with bins of width n
    '''
    cumsum = np.cumsum(x)
    result = cumsum[(n - 1):].copy()
    result[1:] -= cumsum[:-n]
    return result / n


@jit(nopython=True, nogil=True)
def running_mean_arange(start, stop, step, n):
    '''assume start, stop, step as in the input of arange
    given binning of width n
    return the start, stop, step of the resultant arange after binning
    '''
    # middle of the first n bins
    start_avg = start + step * (n - 1) / 2
    # length of the original arange
    N = np.ceil((stop - start) / step)
    stop_avg = start_avg + step * (N - n + 1)  # new length in ()
    return start_avg, stop_avg, step


@jit(nopython=True, nogil=True)
def running_mean_linspace(start, stop, num, n):
    '''assume start, stop, num as in the input of linspace
    given binning of width n
    return the start, stop, num of the resultant arange after binning
    '''
    step = (stop - start) / (num - 1)
    # middle of the first n bins
    mid = step * (n - 1) / 2
    start_avg = start + mid
    stop_avg = stop - mid
    return start_avg, stop_avg, num - n + 1

########################################################################


def get_map_parallel(processes):
    '''return a map function
    uses multiprocessing's Pool if processes != 1
    '''
    if processes == 1:
        return lambda *x: list(map(*x))
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=processes)
        return pool.map


def map_parallel(f, args, mode='multiprocessing', processes=1):
    '''equivalent to map(f, args)
    p: no. of parallel processes when multiprocessing is used
    (in the case of mpi, it is determined by mpiexec/mpirun args)
    mode:

    - mpi: using mpi4py.futures
    - multiprocessing: using multiprocessing from standard library
    - serial: using map
    '''
    if mode == 'mpi':
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            result = executor.map(f, args)
    elif mode == 'multiprocessing' and processes > 1:
        import multiprocessing
        #  with multiprocessing.Pool(processes=processes) as pool: in Python 3
        pool = multiprocessing.Pool(processes=processes)
        try:
            result = pool.map(f, args)
        finally:
            del pool
    else:
        result = list(map(f, args))
    return result


def starmap_parallel(f, args, mode='multiprocessing', processes=1):
    '''equivalent to starmap(f, args)
    p: no. of parallel processes when multiprocessing is used
    (in the case of mpi, it is determined by mpiexec/mpirun args)
    mode:

    - mpi: using mpi4py.futures
    - multiprocessing: using multiprocessing from standard library
    - serial: using starmap
    '''
    if mode == 'mpi':
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            result = executor.starmap(f, args)
    elif mode == 'multiprocessing' and p > 1:
        import multiprocessing
        with multiprocessing.Pool(processes=processes) as pool:
            result = pool.map(lambda x: f(*x), args)
    else:
        from itertools import starmap
        result = list(starmap(f, args))
    return result

# pandas ###############################################################


def insert_index_level(df, level, name, value):
    '''For DataFrame df, add an index with name and value between level & (level + 1)'''
    n = df.index.nlevels
    order = list(range(n))
    order.insert(level, n)

    df = df.copy()
    df[name] = value
    df.set_index(name, append=True, inplace=True)

    return df.reorder_levels(order)


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


def df_unpackbits(data, flag_dict, index=None):
    '''
    data: 1d-array of type int
    flag_dict: a dict with keys as name of the flag and values as the bit of the flag (e.g. 1024)
    similar to unpackbits, but return a DataFrame instead.
    '''
    return pd.DataFrame(
        unpackbits(data, np.array(list(flag_dict.values()), dtype=np.uint64)),
        index=index,
        columns=flag_dict.keys()
    )
