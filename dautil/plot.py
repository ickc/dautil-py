import matplotlib
matplotlib.use('pgf')

from dautil.IO import makedirs
import numpy as np
import os
import pandas as pd
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
    def f_decorated(*args, **kwargs):
        filename = kwargs.pop('filename', None)

        if filename:
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
        elif ext == 'png':
            matplotlib.rcParams.update({"savefig.dpi": 240})

        f(*args, **kwargs)

        if filename:
            if ext == "tikz":
                tikz_save(filename,
                          figureheight='\\figureheight',
                          figurewidth='\\figurewidth')
            else:
                plt.savefig(filename, bbox_inches='tight')
    return f_decorated

# plot pandas DataFrame ################################################


@save
def plot_unique(df, col_select, col_plot, **kwargs):
    '''plot the values of col_plot
    per unique value in col_select
    '''
    for value in df[col_select].unique():
        sns.kdeplot(df[df[col_select] == value][col_plot], label=value, **kwargs)
    plt.title(col_plot)


@save
def plot_unique_index(df, idx_select, col_plot, **kwargs):
    '''plot the values of col_plot
    per index from index level idx_select
    '''
    indexslice = [slice(None)] * df.index.nlevels

    for value in df.index.levels[idx_select]:
        # choose the current value
        indexslice[idx_select] = slice(value)
        sns.kdeplot(df.loc[tuple(indexslice), :][col_plot], label=value, **kwargs)

    plt.title(col_plot)


@save
def plot_unique_index_binned(df, idx_select, col_plot, binwidth, **kwargs):
    '''similar to plot_unique_index, but the values of index in level idx_select
    is binned instead
    Assumed df is sorted, else its behavior is undefined.

    Examples
    --------
    >>> plot_unique_index_binned(df_masked, 0, 'ampparams2_1', binwidth=pd.Timedelta('10 days'))
    '''
    indexslice = [slice(None)] * df.index.nlevels

    values = df.index.levels[idx_select]
    n = values.shape[0]
    idx_current = values[0]
    idx_end = values[-1]

    while idx_current <= idx_end:
        idx_new = idx_current + binwidth
        # choose the current value
        indexslice[idx_select] = slice(idx_current, idx_new)
        try:
            sns.kdeplot(df.loc[tuple(indexslice), :][col_plot], label=idx_current, **kwargs)
        except (ValueError, ZeroDivisionError):
            print('No data between {} and {}.'.format(idx_current, idx_new))
        idx_current = idx_new

    plt.title(col_plot)


@save
def plot_corr(df, vmin=0., mask=None, triangle=True, **kwargs):
    '''Assume df is a correlation matrix
    if triangle, plot the lower-half triangle excluding diagonal only.
    mask will be passed to seaborn's heatmap, data will not be shown in cells where ``mask`` is True.
    '''
    fig = plt.figure(**kwargs)

    # Generate a mask for the upper triangle
    if triangle:
        mask_local = np.zeros_like(df, dtype=np.bool)
        mask_local[np.triu_indices_from(mask_local)] = True
        if mask is None:
            mask = mask_local
        else:
            mask |= mask_local
        del mask_local

    sns.heatmap(df, vmin=vmin, mask=mask, square=True)

########################################################################


@save
def plot_camb(df, **kwargs):
    '''plot DataFrame obtained by load_camb
    '''
    return df.plot(sharex=True, subplots=True, **kwargs)


@save
def plot_x_y_complex(x, y, **kwargs):
    '''x is a real array.
    y is a complex array.
    '''
    plt.figure()
    sns.tsplot(x).set_title('x', **kwargs)
    plt.figure()
    sns.tsplot(np.real(y)).set_title('real(y)', **kwargs)
    plt.figure()
    sns.tsplot(np.imag(y)).set_title('imag(y)', **kwargs)

    sns.jointplot(x=x, y=np.real(y), kind='reg', **kwargs)
    plt.title('real(y) vs. x')
    sns.jointplot(x=x, y=np.imag(y), kind='reg', **kwargs)
    plt.title('imag(y) vs. x')
    sns.jointplot(x=x, y=np.absolute(y), kind='reg', **kwargs)
    plt.title('abs(y) vs. x')
    sns.jointplot(x=x, y=np.angle(y), kind='reg', **kwargs)
    plt.title('arg(y) vs. x')

# PIL ##################################################################

def array_to_image(array, filename, text='', fontname='lmsans12-regular.otf', font_ratio=10):
    '''array: dtype of numpy.float64
    filename: output filename
    text: optional text to print on top-left corner
    fontname: fontname that's available on the system for ``text``
    font_ratio: the ratio of the font-size comparing to horizontal size of image
    '''
    from PIL import Image

    # transpose and invert y-axis
    img = Image.fromarray((array.T)[::-1])
    if text:
        from PIL import ImageFont, ImageDraw
        draw = ImageDraw.Draw(img)
        fontsize = array.shape[0] // font_ratio
        font = ImageFont.truetype(fontname, fontsize)
        draw.text((0, 0), text,(255,), font=font)
    img.save(filename)
