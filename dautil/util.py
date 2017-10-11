import numpy as np


def summarize(data):
    '''If data is dict, return a dict with values of types of values in data
    else return type
    if the type is numpy.ndarray, instead of showing type, show dtype and shape
    '''
    if not isinstance(data, dict):
        return type(data)
    summarized = {}
    for i, j in data.items():
        if isinstance(j, dict):
            summarized[i] = summarize(j)
        elif isinstance(j, np.ndarray):
            summarized[i] = (j.dtype, j.shape) 
        else:
            summarized[i] = type(j)
    return summarized


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
