#!/usr/bin/env python

'''Assert HDF5 input is non-zero.
Print to stderr if not.
For example,

find . -iname '*.hdf5' -exec h5zero.py {} +
'''

from __future__ import print_function

import argparse
import h5py
import numpy as np
import sys

__version__ = '0.1'


def assert_hdf5_nonzero(f, verbose=False):
    if isinstance(f, h5py._hl.dataset.Dataset):
        if verbose:
            print('{} is dataset, asserting...'.format(f))
        temp = np.nan_to_num(f)
        assert temp.any()
    elif isinstance(f, h5py._hl.group.Group):
        if verbose:
            print('{} is group, entering...'.format(f))
        for i in f:
            assert_hdf5_nonzero(f[i], verbose)


def main(args):
    for filename in args.input:
        with h5py.File(filename, "r") as f:
            try:
                assert_hdf5_nonzero(f, verbose=args.verbose)
            except AssertionError:
                print(filename, file=sys.stderr)


def cli():
    parser = argparse.ArgumentParser(description='Assert HDF5 input is non-zero.')
    parser.set_defaults(func=main)

    # define args
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('input', nargs='+',
                        help='Input HDF5 files. Can be more than 1.')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='verbose to stdout.')

    # parsing and run main
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
