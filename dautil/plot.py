import matplotlib
matplotlib.use('pgf')

from dautil.IO import makedirs
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
# from matplotlib2tikz import save as tikz_save


def save(f):
    '''a decorator to add keyword filename to the function args
    the function f is expected to be a procedure that uses matplotlib
    the figure produced is saved to the path specified in filename
    dispatch according to the filename extension
    '''
    @wraps(f)
    def f_decorated(*args, filename=None, **kwargs):
        makedirs(os.path.dirname(filename))

        ext = os.path.splitext(filename)[1][1:] if filename else None

        if ext in ('pdf', 'pgf'):
            # setup fonts using Latin Modern
            matplotlib.rcParams.update({
                "font.family": ["serif"],
                "font.serif": ["Latin Modern Roman"],
                "font.sans-serif": ["Latin Modern Sans"],
                "font.monospace": ["Latin Modern Mono"]
            })

        f(*args, **kwargs)

        if filename:
            if ext == "tikz":
                tikz_save(filename,
                          figureheight='\\figureheight',
                          figurewidth='\\figurewidth')
            else:
                plt.savefig(filename)
    return f_decorated


def plot_x_y_complex(x, y):
    '''x is a real array.
    y is a complex array.
    '''
    plt.figure()
    sns.tsplot(x).set_title('x')
    plt.figure()
    sns.tsplot(np.real(y)).set_title('real(y)')
    plt.figure()
    sns.tsplot(np.imag(y)).set_title('imag(y)')

    sns.jointplot(x=x, y=np.real(y), kind='reg')
    plt.title('real(y) vs. x')
    sns.jointplot(x=x, y=np.imag(y), kind='reg')
    plt.title('imag(y) vs. x')
    sns.jointplot(x=x, y=np.absolute(y), kind='reg')
    plt.title('abs(y) vs. x')
    sns.jointplot(x=x, y=np.angle(y), kind='reg')
    plt.title('arg(y) vs. x')
