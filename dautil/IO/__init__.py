import os
import pickle
import sys

PY2 = sys.version_info[0] == 2


def _read_pkl_helper(path, encoding='ASCII'):
    '''read pkl from path
    '''
    with open(path, 'rb') as f:
        return pickle.load(f, encoding=encoding)


def read_pkl2(path):
    '''read pkl saved in py2.
    '''
    encoding = 'ASCII' if PY2 else 'latin1'
    return _read_pkl_helper(path, encoding=encoding)


def read_pkl(path):
    '''read pkl from f, unsure if it was saved in py2 or py3.
    '''
    try:
        return _read_pkl_helper(path)
    except UnicodeDecodeError:
        return _read_pkl_helper(path, encoding='latin1')


def _read_pkl_all_iter_helper(path, encoding='ASCII'):
    '''read all pkl from path, unsure if it was saved in py2 or py3.

    this one will keep reading until EOF
    '''
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f, encoding=encoding)
            except EOFError:
                break


def read_pkl_all(path):
    '''read all pkl from path, unsure if it was saved in py2 or py3.

    this one will keep reading until EOF
    '''
    try:
        return list(_read_pkl_all_iter_helper(path))
    except UnicodeDecodeError:
        return list(_read_pkl_all_iter_helper(path, encoding='latin1'))


def read_h5_dataset(path, dataset):
    '''a functional thin wrapper to read hdf5 dataset
    from path
    '''
    import h5py

    with h5py.File(path, 'r') as f:
        return f[dataset][:]


def makedirs(path):
    '''makesdirs if not exist while avoiding race condition
    catch the case that path is file, whether initially or in a race condition
    '''
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
