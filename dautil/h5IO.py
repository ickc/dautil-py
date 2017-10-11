from __future__ import print_function

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def h5delete(filename, dry_run=True, verbose=False):
    try:
        with h5py.File(filename, "r") as f:
            if verbose:
                print(filename, 'is good.')
    except IOError:
        if verbose:
            print(filename, 'is not good, and will be deleted.', file=sys.stderr)
        else:
            print(filename, file=sys.stderr)
        if not dry_run:
            os.remove(filename)


def h5assert_nonzero(f, verbose=False):
    if isinstance(f, h5py._hl.dataset.Dataset):
        if verbose:
            print('{} is dataset, asserting...'.format(f))
        temp = np.nan_to_num(f)
        assert temp.any()
    elif isinstance(f, h5py._hl.group.Group):
        if verbose:
            print('{} is group, entering...'.format(f))
        for i in f:
            h5assert_nonzero(f[i], verbose)


def h5assert(f1, f2, rtol=1.5e-09, atol=1.5e-09, verbose=False):
    if isinstance(f1, h5py._hl.dataset.Dataset):
        if verbose:
            print('{}, {} are dataset, asserting...'.format(f1, f2))
        temp1 = np.nan_to_num(f1)
        temp2 = np.nan_to_num(f2)
        np.testing.assert_allclose(temp1, temp2, rtol, atol)
    elif isinstance(f1, h5py._hl.group.Group):
        if verbose:
            print('{} is group, entering...'.format(f1))
        for i in f1:
            try:
                h5assert(f1[i], f2[i], rtol, atol, verbose)
            except KeyError:
                raise AssertionError


def plot_h5diff(f1, f2, out_dir, prefix='', verbose=False):
    if isinstance(f1, h5py._hl.dataset.Dataset):
        name = '-'.join([prefix] + f1.name.split('/')[1:])
        if verbose:
            print('{} is dataset, plotting...'.format(name))
        temp1 = np.nan_to_num(f1)
        temp2 = np.nan_to_num(f2)
        temp3 = np.abs(temp2 - temp1)
        # plt.plot(temp1)
        # plt.plot(temp2)
        plt.plot(temp3)
        plt.savefig(os.path.join(out_dir, name + '.png'))
        plt.close()
    elif isinstance(f1, h5py._hl.group.Group):
        if verbose:
            name = '-'.join([prefix] + f1.name.split('/')[1:])
            print('{} is group, entering...'.format(name))
        for i in f1:
            plot_h5diff(f1[i], f2[i], out_dir, prefix, verbose)


def h5split(in_file, out_dir, verbose=False):
    '''split each of the HDF5 group from in_file to individual ones in out_dir
    '''
    filename = os.path.splitext(os.path.basename(in_file))
    with h5py.File(in_file, "r") as f_in:
        for group in f_in:
            out_file = os.path.join(out_dir, filename[0] + '_' + group + filename[1])
            if verbose:
                print(out_file)
            with h5py.File(out_file, "x") as f_out:
                for sub_group in f_in[group]:
                    h5_path = group + '/' + sub_group
                    if verbose:
                        print(h5_path)
                    f_in.copy(h5_path, f_out)
