import pickle


def read_pkl2(path):
    '''read pkl saved in py2.
    '''
    with open(path, 'rb') as f:
        return pickle.load(f) if PY2 else pickle.load(f, encoding='latin1')
