import numpy as np
import pandas as pd


def load_camb(path):
    '''load CAMB output dat as a pandas DataFrame.
    Assume the columns are ('l', 'TT', 'EE', 'BB', 'TE')
    '''
    return pd.read_csv(path,
                       dtype={'l': np.int32, 'TT': np.float32, 'EE': np.float32, 'BB': np.float32, 'TE': np.float32},
                       delim_whitespace=True,
                       header=None,
                       index_col=0,
                       names=('l', 'TT', 'EE', 'BB', 'TE'))
