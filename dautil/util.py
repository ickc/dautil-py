from __future__ import print_function

import operator
import sys
import types
from builtins import map
from collections import defaultdict
from functools import reduce, wraps

import numba
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from numba import jit

PY2 = sys.version_info[0] == 2


def summarize_types(data):
    '''return a set of all types within a dict/iterable

    :param data: possibly nested dict/list/tuple structure

    :return: set of types of values inside `data`
    :rtype: set
    '''
    if isinstance(data, dict):
        return set().union(*(get_types(value) for value in data.values()))
    elif isinstance(data, (list, tuple)):
        return set().union(*(get_types(value) for value in data))
    else:
        return {type(data)}

# summarize ############################################################


def summarize_ndarray(data):
    '''assume data is ndarray
    return its type, dtype and shape'''
    return (type(data), data.dtype, data.shape)


def flatten_list(data, verbose=False):
    '''flatten nested list/tuple to arbitrary depth

    Algorithm:

    keep flatten and data list
    initially flatten list is empty
    moving data list to flatten list until
    data list is empty

    The followings are equivalent, but limited by recursions
    ```py
    def flatten_list(data):
        result = []
        for datum in data:
            if isinstance(datum, (list, tuple)):
                result += flatten_list(datum)
            else:
                result.append(datum)
        return result


    def flatten_list(data):
        return [item
                    for datum in data
                    for item in (flatten(datum) if isinstance(datum, list) else [datum])
               ]
    ```
    '''
    flatten = []
    list_found = True
    while list_found:
        list_found = False
        for i, datum in enumerate(data):
            if isinstance(datum, (list, tuple)):
                list_found = True

                head = data[:i]
                tail = data[i + 1:]

                flatten += list(head)
                data = list(datum) + list(tail)

                if verbose:
                    print('''I found a list at position {}.
Moving {} to flatten list, adding {} and {} as the remaining data.
My current flatten list is {} and data is {}.'''.format(
                        i,
                        head,
                        datum,
                        tail,
                        flatten,
                        data
                    ))

                break

        if not list_found:
            flatten += data
            if verbose:
                print('''All remaining data is flatten, adding it to flatten
My current flatten list is {} and data is {}'''.format(flatten, data))
    return flatten


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


def summarize_dicts(dicts):
    '''``dicts``: iterable of dict
    per key per dict, form a set of possible types across all element of ``dicts``
    This is useful to inspect a bunch of dict sharing the same structure with slightly
    different types (e.g. NoneType or list)
    '''
    result = defaultdict(set)
    for dict_ in dicts:
        for key, value in dict_.items():
            result[key].add(type(value))
    return result


def get_variables(module):
    '''Return a list of variable names from ``module``.
    Currently, callables and modules are ignored.
    '''
    return [item for item in dir(module) if not
            (item.startswith("__") or
             isinstance(getattr(module, item), types.ModuleType)
             or callable(getattr(module, item)))]


def assert_dict(input1, input2, rtol=1.5e-09, atol=1.5e-09, verbose=False):
    '''
    recursively assert into a dictionary
    if the value is a numpy array, assert_allclose, else assert equal.
    '''
    for key in input1:
        if isinstance(input1[key], dict):
            if verbose:
                print('asserting {}'.format(key))
            assert_dict(input1[key], input2[key], rtol=rtol, atol=atol, verbose=verbose)
        elif isinstance(input1[key], np.ndarray):
            if verbose:
                print('asserting {}'.format(key))
            np.testing.assert_allclose(input1[key], input2[key], rtol=rtol, atol=atol)
        else:
            if verbose:
                print('asserting {}'.format(key))
            assert input1[key] == input2[key]


def assert_arrays(arrays, assert_func=np.testing.assert_array_equal):
    '''arrays: an iterable of numpy.ndarray
    this function assert each of them are equal
    assert_func can be defined to use other assertions from numpy
    beware that all other arrays are comparing to the first array
    '''
    # make sure it is an iterator
    arrays = iter(arrays)

    array0 = next(arrays)
    for array in arrays:
        assert_func(array0, array)


def remove_key(data, key, level=1):
    '''``data``: dictionary. If ``level`` > 1, assumes
    it is a dictionary of dictionaries of at least this much level.
    This function remove the key of the dictionary at this level.
    ``level``: 1 is interpreted as the values of the dictionary ``data``.
    2 is interpreted as the values of the dictionary which is a value of ``data``,
    so on so forth.
    '''
    if level == 1:
        del data[key]
    else:
        for value in data.values():
            remove_key(value, key, level=(level - 1))


def sum_(*args):
    '''first argument: ``iterable``.

    second argument: ``start`` (default: 0)

    A drop-in replacement of ``sum`` with arbitrary type.
    The only requirement is the type of elements in ``iterable``
    has `__iadd__` method and is compatible with each other.
    ``start`` is not used except for empty iterable in which case it
    is returned.
    '''
    from copy import deepcopy

    iterable = iter(args[0])
    try:
        # avoid mutating the input, both during and after this function call
        result = deepcopy(next(iterable))
        for i in iterable:
            result += i
        return result
    except StopIteration:
        start = args[1] if len(args) > 1 else 0
        return start


def product_(*args):
    '''first argument: ``iterable``.

    second argument: ``start`` (default: 0)

    similar to ``sum_`` but reduce by product
    '''
    from copy import deepcopy

    iterable = iter(args[0])
    try:
        # avoid mutating the input, both during and after this function call
        result = deepcopy(next(iterable))
        for i in iterable:
            result *= i
        return result
    except StopIteration:
        start = args[1] if len(args) > 1 else 0
        return start


def split_list(list_, n):
    '''split list into n chunks
    first n - 1 elements are of equal length
    and last element is the remainder
    '''
    N = len(list_)
    n_per_chunk = (N + n - 1) // n
    return [list_[i:i + n_per_chunk] for i in range(0, N, n_per_chunk)]

# compose ##############################################################


def compose(*functions):
    '''composing functions. For example,

    compose(f, g, h)(x) == f(g(h(x)))
    '''
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

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


def filter_box(array):
    '''filter out empty edges of a 2d-array
    '''
    box = get_box(array)
    return array[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1]]


def mask(f, idxs=(0,)):
    '''f: a function with args where the idx-th arguments are numpy.ndarray
    a decorator to add keyword mask of type boolean array
    if mask is not None, mask the first positional argument
    the returned value will be "unmasked", with default value zeros
    TODO: add options for other default values
    '''
    @wraps(f)
    def f_decorated(*args, **kwargs):
        mask = kwargs.pop('mask', None)

        if mask is None:
            return f(*args, **kwargs)
        else:
            args = [arg[mask] if idx in idxs else arg for idx, arg in enumerate(args)]

            temp = f(*args, **kwargs)

            result = np.zeros_like(mask, dtype=temp.dtype)
            result[mask] = temp
            return result

    return f_decorated


@mask
def to_levels(array, dtype=np.uint8, ranges=(0, 255), fix_origin=False):
    '''array: of type numpy.ndarray
    mask: optional mask when scaling ``array``
    dtype: output dtype
    ranges: a tuple of 2 values, defining the range of values the input ``array`` is going to be
    mapped to.
    fix_origin: if True, 0 stays 0
    '''
    _min = array.min()
    _max = array.max()
    if fix_origin:
        rs = [range_i / value for range_i, value in zip(ranges, (_min, _max)) if (range_i and value)]
        # if the range of array includes 0
        scale = (min if _min * _max < 0. else max)(*rs) if len(rs) == 2 else (rs[0] if len(rs) == 1 else 0.)
        del rs
        result = array * scale
    else:
        scale = (ranges[1] - ranges[0]) / ((_max - _min) or 1.)
        result = array * scale + (ranges[0] - _min * scale)
    return result.astype(dtype)


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


def running_mean_axis(xs, n, axis=0):
    '''
    axis: must be positive
    return: array of the moving average of x with bins of width n along given axis
    '''
    cumsums = xs.cumsum(axis)
    results = cumsums[(slice(None),) * axis + (slice(n - 1, None, None),)].copy()
    results[(slice(None),) * axis + (slice(1, None, None),)] -= cumsums[(slice(None),) * axis + (slice(None, -n, None),)]
    return results / n


@jit(nopython=True, nogil=True)
def running_mean(x, n):
    '''
    special case of ``running_mean_axis`` for 1d-array
    '''
    cumsum = x.cumsum()
    result = cumsum[(n - 1):].copy()
    result[1:] -= cumsum[:-n]
    return result / n


@jit(nopython=True, nogil=True)
def polynomials(x, k):
    '''``x``: 1d-array
    ``k``: maximum order
    return
    ------
    2d-array, where each column corresponse to x**i, i from 1 to k
    '''
    n = x.size
    result = np.empty((n, k))
    for i in range(n):
        temp = x[i]
        result[i, 0] = temp
        for j in range(1, k):
            result[i, j] = temp * result[i, j - 1]
    return result


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


@numba.vectorize([numba.float64(numba.complex128, numba.complex128), numba.float32(numba.complex64, numba.complex64)])
def complex_dot(x, y):
    '''return the complex dot product of ``x``, ``y``
    '''
    return x.real * y.real + x.imag * y.imag


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    '''return the square norm of complex ``x``
    equals to complex_dot(x, x)
    '''
    real = x.real
    imag = x.imag
    return real * real + imag * imag


@jit(nopython=True, nogil=True, parallel=True)
def min_offdiag(array):
    '''``array``: 2d-array
    return 1d-array of the minimum per row excluding the diagonal.
    Note that the result doesn't make sense if array is of shape (1, 1)
    same as min_offdiag_general(array, axis=1) but faster
    '''
    n = array.shape[0]
    result = np.empty(n, dtype=array.dtype)
    for i in numba.prange(n):
        _min = np.inf
        for j in range(n):
            if j != i:
                temp = array[i, j]
                if temp < _min:
                    _min = temp
        result[i] = _min
    return result


def min_offdiag_general(array, **kwargs):
    '''works similar to ``np.amin(array, **kwargs)``
    except that the off-diagonal items is ignored when finding minimum
    '''
    temp = array.copy()
    np.fill_diagonal(temp, np.nan)
    return np.nanmin(temp, **kwargs)


@jit(nopython=True, nogil=True, parallel=True)
def max_offdiag(array):
    '''``array``: 2d-array
    return 1d-array of the maximum per row excluding the diagonal.
    Note that the result doesn't make sense if array is of shape (1, 1)
    same as max_offdiag_general(array, axis=1) but faster
    '''
    n = array.shape[0]
    result = np.empty(n, dtype=array.dtype)
    for i in numba.prange(n):
        _max = np.NINF
        for j in range(n):
            if j != i:
                temp = array[i, j]
                if temp > _max:
                    _max = temp
        result[i] = _max
    return result


def max_offdiag_general(array, **kwargs):
    '''works similar to ``np.amax(array, **kwargs)``
    except that the off-diagonal items is ignored when finding maximum
    '''
    temp = array.copy()
    np.fill_diagonal(temp, np.nan)
    return np.nanmax(temp, **kwargs)


@jit(nopython=True, nogil=True, parallel=True)
def reciprocal_sum_reciprocal(*args):
    '''reciprocal of sum of reciprocal of args
    ``len(args) > 1`` for this to make sense
    '''
    result = np.reciprocal(args[0])
    for arg in args[1:]:
        result += np.reciprocal(arg)
    return np.reciprocal(result)


@jit(nopython=True, nogil=True)
def zero_padding_idx(in_shape, out_shape):
    '''``in_shape``: input 2d-array shape
    ``out_shape``: output 2d-array shape

    return i_min, i_max, j_min, j_max that defines
    the box within the output array to be copied from
    the input array
    '''
    m, n = in_shape
    i = (out_shape[0] - m) // 2
    j = (out_shape[1] - n) // 2

    return i, i + m, j, j + n


@jit(nopython=True, nogil=True)
def zero_padding(mask, shape):
    '''``mask``: 2d-array
    zero-padding ``mask`` to target shape
    '''
    result = np.zeros(shape, dtype=mask.dtype)
    i_min, i_max, j_min, j_max = zero_padding_idx(mask.shape, shape)
    result[i_min:i_max, j_min:j_max] = mask
    return result


@jit(nopython=True, nogil=True)
def planck_taper(m):
    '''return a planck taper window, mapping from 0..m to 0..1,
    excluding 0, m index (which equals to 0, and 1 respectively)
    i.e. the returned array has length m - 1
    the reason is that the array is assumed to be zero-padded anyway,
    where the 0 at the boundary is granted.
    '''
    n = np.arange(1, m)
    return 1. / (1. + np.exp(m * ((m - 2. * n) / (n * (m - n)))))


@jit(nopython=True, nogil=True, parallel=True)
def clamp_edge(mask, width):
    '''taper ``mask`` with ``width`` in-place,
    using ``planck_taper``
    assume mask's size bigger than 2 * width
    in-place in ``mask``
    '''
    taper = planck_taper(width + 1)
    m, n = mask.shape

    # apply taper
    # assume mask is much larger than width
    # i.e. cache favor mask first
    # i-lower
    for i in range(width):
        mask[i] *= taper[i]
    # j-lower
    for i in range(m):
        for j in range(width):
            mask[i, j] *= taper[j]
    # i-upper
    for i in range(width - 1, -1, -1):
        mask[m - i - 1] *= taper[i]
    # j-upper
    for i in range(m):
        for j in range(width - 1, -1, -1):
            mask[i, n - j - 1] *= taper[j]


@jit(nopython=True, nogil=True)
def get_non_nan_idxs(array):
    '''return the idxs that is not nan
    '''
    idxs_non_nan = np.nonzero(~np.isnan(array))[0]
    return idxs_non_nan

# KDE ##################################################################


def get_KDE(data, num=100, **kwargs):
    '''given a distribution ``data``,
    return the KDE estimation as a 2d-array,
    where the columns are the x, y values repsectively

    ``num``: num of rows of the resultant array
    ``kwargs``: passes to ``scipy.stats.gaussian_kde
    '''
    func = scipy.stats.gaussian_kde(data, **kwargs)
    x = np.linspace(data.min(), data.max(), num=num)
    return np.column_stack((x, func(x)))


def get_KDE_der(data, num=100, **kwargs):
    '''Similar to ``get_KDE``, but return the first
    derivative of it instead.
    '''
    func = scipy.stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), num=num)
    dx = x[1] - x[0]
    return np.column_stack((x, scipy.misc.derivative(func, x, dx=dx)))


def get_KDE_min(data, num=100, **kwargs):
    '''given a distribution ``data``,
    return the minimum of the KDE estimation as a 2d-array.

    ``num``: number of points between data.min() and data.max(),
    this controls the resolution of the minimum
    ``kwargs``: passes to ``scipy.stats.gaussian_kde

    This is useful for clustering based on distribution.
    '''
    xy = get_KDE(data, num=num, **kwargs)
    return xy[:, 0][scipy.signal.argrelmin(xy[:, 1])]


def get_KDE_der_max(data, num=100, **kwargs):
    '''Similar to ``get_KDE_min``, but
    get the maximum of the first derivatives
    of the Gaussian KDE.

    This can be useful when ``get_KDE_min`` provides no
    minimum.
    '''
    xy = get_KDE_der(data, num=num, **kwargs)
    return xy[:, 0][scipy.signal.argrelmax(xy[:, 1])]


def min_nan(array, **kwargs):
    '''same as ``numpy.amin(array, **kwargs)``
    except if the array is empty, it returns ``numpy.nan`` instead.
    '''
    try:
        return np.amin(array, **kwargs)
    # for empty array
    except ValueError:
        return np.nan

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


def _starmap(f, x):
    '''return f(*x), for map_parallel only
    '''
    return f(*x)


def map_parallel(f, *args, **kwargs):
    '''equivalent to map(f, *args)
    processes: no. of parallel processes when multiprocessing is used
    (in the case of mpi, it is determined by mpiexec/mpirun args)
    mode:

    - mpi: using mpi4py.futures
    - multiprocessing: using multiprocessing from standard library
    - multithreading: using multithreading from standard library
    - serial: using map

    Note: this is for Python 2 compatibility,
    else map_parallel(f, *args, mode='multiprocessing', processes=1)
    '''
    mode = kwargs.get('mode', 'multiprocessing')
    processes = kwargs.get('processes', 1)

    if mode == 'mpi':
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            result = executor.map(f, *args)
    elif mode == 'multiprocessing' and processes > 1:
        if PY2:
            import multiprocessing
            from functools import partial
            pool = multiprocessing.Pool(processes=processes)
            try:
                result = pool.map(f, *args) \
                    if len(args) <= 1 else \
                    pool.map(partial(_starmap, f), zip(*args))
            finally:
                pool.terminate()
                pool.close()
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=processes) as executor:
                result = list(executor.map(f, *args))
    elif mode == 'multithreading' and processes > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=processes) as executor:
            result = list(executor.map(f, *args))
    else:
        result = list(map(f, *args))
    return result


def starmap_parallel(f, args, mode='multiprocessing', processes=1):
    '''equivalent to starmap(f, args)
    processes: no. of parallel processes when multiprocessing is used
    (in the case of mpi, it is determined by mpiexec/mpirun args)
    mode:

    - mpi: using mpi4py.futures
    - multiprocessing: using multiprocessing from standard library
    - multithreading: using multithreading from standard library
    - serial: using starmap
    '''
    if mode == 'mpi':
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            result = executor.starmap(f, args)
    elif mode == 'multiprocessing' and processes > 1:
        from functools import partial
        if PY2:
            import multiprocessing
            pool = multiprocessing.Pool(processes=processes)
            try:
                result = pool.map(partial(_starmap, f), args)
            finally:
                pool.terminate()
                pool.close()
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=processes) as executor:
                result = list(executor.map(partial(_starmap, f), args))
    elif mode == 'multithreading' and processes > 1:
        from functools import partial
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=processes) as executor:
            result = list(executor.map(partial(_starmap, f), args))
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
        def _linregress(x): return scipy.stats.linregress(x.reset_index(level=regresslevel).values)
    elif regressorder == 1:
        def _linregress(x): return scipy.stats.linregress(x.reset_index(level=regresslevel).values)[regressindex]
    else:
        def _linregress(x): return scipy.stats.linregress(x.reset_index(level=regresslevel).values)[regressindex]**regressorder

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
        unpackbits(data, np.array(list(flag_dict.values()))),
        index=index,
        columns=flag_dict.keys()
    )
