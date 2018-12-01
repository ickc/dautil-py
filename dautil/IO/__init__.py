import os
import pickle
import sys

PY2 = sys.version_info[0] == 2


def read_pkl2(path):
    '''read pkl saved in py2.
    '''
    with open(path, 'rb') as f:
        return pickle.load(f) if PY2 else pickle.load(f, encoding='latin1')

def read_pkl(path):
    '''read pkl, unsure if it was saved in py2 or py3.
    '''
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            return pickle.load(f, encoding='latin1')


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
