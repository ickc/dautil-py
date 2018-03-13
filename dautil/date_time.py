from datetime import datetime
import numpy as np

# converters ###########################################################

def strptime_np_general(string, fmt='%Y%m%d_%H%M%S'):
    '''
    string: a datetime in format ``fmt``
    return: numpy.datetime64 representation of it
    '''
    return np.datetime64(
        datetime.strptime(string, fmt),
        's'
    )


def strptime_np(string):
    '''
    string: a datetime in format '%Y%m%d_%H%M%S'
    return: numpy.datetime64 representation of it
    equiv. to strtime_np_general
    but faster by over an order of magnitude
    '''
    return np.datetime64(
        string[:4] + '-' + string[4:6] + '-' + string[6:8] + 'T' + string[9:11] + ':' + string[11:13] + ':' + string[13:15]
    )


def npptime(datetime64):
    '''
    datetime64: of type numpy.datetime64
    return: Python's datetime format of it
    '''
    return datetime.utcfromtimestamp(datetime64.astype(int))


def strttime_np_general(datetime64, fmt='%Y%m%d_%H%M%S'):
    '''
    datetime64: of type numpy.datetime64
    return: datetime in format ``fmt``
    '''
    return npptime(datetime64).strftime(fmt)



def strttime_np(datetime64):
    '''
    datetime64: of type numpy.datetime64
    return: datetime in format '%Y%m%d_%H%M%S'
    equiv. to datetime.utcfromtimestamp(datetime64.astype(int)).strftime('%Y%m%d_%H%M%S')
    but faster by almost an order of magnitude
    '''
    temp = str(datetime64)
    return temp[:4] + temp[5:7] + temp[8:10] + '_' + temp[11:13] + temp[14:16] + temp[17:19]
