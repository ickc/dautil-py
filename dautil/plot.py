import os
from functools import wraps

import holoviews as hv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dautil.IO import makedirs


def sns_heatmap_xy(mask, **kwargs):
    '''``mask``: 2d-array
    thin wrapper of seaborn.heatmap to plot a 2d-array
    like an x-y plot.
    such that the horizontal axis is the first axis,
    increasing from left to right;
    and the vertical axis is the second axis,
    increasing from bottom to top.
    '''
    return sns.heatmap(mask.T, **kwargs).invert_yaxis()


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
                from matplotlib2tikz import save as tikz_save
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
        sns.kdeplot(
            df[df[col_select] == value][col_plot],
            label=value,
            **kwargs
        )
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
        sns.kdeplot(
            df.loc[tuple(indexslice), :][col_plot],
            label=value,
            **kwargs
        )

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
    idx_current = values[0]
    idx_end = values[-1]

    while idx_current <= idx_end:
        idx_new = idx_current + binwidth
        # choose the current value
        indexslice[idx_select] = slice(idx_current, idx_new)
        try:
            sns.kdeplot(
                df.loc[tuple(indexslice), :][col_plot],
                label=idx_current,
                **kwargs
            )
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
        draw.text((0, 0), text, (255,), font=font)
    img.save(filename)

# plot.ly


def iplot_column_slider(df, active=0):
    '''similar to cufflinks' iplot method
    but apply a slider scross the columns so that only 1 column
    is shown at a time

    returns a figure object to be consumed by iplot

    Example
    -------

    ```py
    import plotly.offline as py
    py.init_notebook_mode()
    py.iplot(iplot_column_slider(df))
    ```
    '''
    is_multi = isinstance(df.columns, pd.core.indexes.multi.MultiIndex)

    data = [
        {
            'visible': False,
            'name': ', '.join(map(str, col)) if is_multi else str(col),
            'x': series.index,
            'y': series.values
        }
        for col, series in df.items()
    ]
    data[active]['visible'] = True

    n = df.shape[1]
    steps = [
        {
            'method': 'restyle',
            'args': [
                'visible',
                [False] * i + [True] + [False] * (n - i - 1)
            ],
            'label': ', '.join(map(str, col)) if is_multi else str(col)
        }
        for i, col in enumerate(df.columns)
    ]

    sliders = [{
        'active': active,
        'currentvalue': {'prefix': 'Column: '},
        'pad': {'t': 50},
        'steps': steps,
    }]

    return {
        'data': data,
        'layout': {'sliders': sliders, 'showlegend': True}
    }


def plot_column_slider(df, chart=hv.Curve, slider=False, imag_label='error'):
    '''create a Holoviews dynamic map that slice through each column

    `df`: DataFrame. if complex, take the real part as the value and imaginary part as the error bar

    `chart`: any Holoview Chart class such as Curve, Scatter

    `slider`: if True, force slider

    `imag_label`: label for the curve in the imaginary part

    hint: add ``%%opts Curve {+framewise}`` to readjust the frame on each selection
    '''
    # dispatch MultiIndex or not
    is_multi = isinstance(df.columns, pd.core.indexes.multi.MultiIndex)
    is_complex = np.iscomplexobj(df)

    def plot(*args):
        # get series from a slice and its label
        if is_multi:
            _args = [
                levels[arg]
                for levels, arg in zip(df.columns.levels, args)
            ] if slider else args
            _slice = tuple(map(lambda x: slice(x, x), _args))
            series = df.loc[:, _slice]
            label = ', '.join(map(str, _args))
            del _args, _slice
        # since it is not a MultiIndex, args is of length 1
        else:
            arg = args[0]
            _arg = df.columns[arg] if slider else arg
            series = df[_arg]
            label = str(_arg)
            del arg, _arg

        x = series.index
        y = series.values
        del series

        if is_complex:
            _chart = chart(np.column_stack((x, y.real)), label=label)
            _chart *= chart(np.column_stack((x, y.imag)), label=imag_label)
        else:
            _chart = chart(np.column_stack((x, y)), label=label)

        if slider:
            _chart = _chart.opts(title=label)
        return _chart

    dmap = hv.DynamicMap(plot, kdims=df.columns.names)

    values = dict(zip(df.columns.names, df.columns.levels)) \
        if is_multi else \
        {df.columns.name: df.columns.values}
    if slider:
        values = {name: range(len(value)) for name, value in values}

    return dmap.redim.values(**values)
