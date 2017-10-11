import h5py
import numpy as np
import os
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
