import os
import pickle
import sys

PY2 = sys.version_info[0] == 2


def read_pkl2(path):
    '''read pkl saved in py2.
    '''
    with open(path, 'rb') as f:
        return pickle.load(f) if PY2 else pickle.load(f, encoding='latin1')


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
