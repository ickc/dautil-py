#!/usr/bin/env python

'''
For example,

find small-test -iname '*.hdf5' | sed 's/^small-test\(.*\)$/h5diff\.py -1 small-test\1 -2 small-test-reference\1 -r 1e-1 -a 1 -S/g' | xargs -i -n1 -P1 bash -c '$0' {}
'''

from __future__ import print_function

import argparse
import h5py
import numpy as np
import sys

__version__ = '0.1'


def assert_hdf5(f1, f2, rtol=1.5e-09, atol=1.5e-09, verbose=False):
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
                assert_hdf5(f1[i], f2[i], rtol, atol, verbose)
            except KeyError:
                raise AssertionError


def main(args):
    with h5py.File(args.first, "r") as f1:
        with h5py.File(args.second, "r") as f2:
            try:
                assert_hdf5(f1, f2, args.rtol, args.atol, verbose=args.verbose)
            except AssertionError:
                if args.silent:
                    print(args.first, file=sys.stderr)
                else:
                    raise


def cli():
    parser = argparse.ArgumentParser(description='Compare if 2 HDF5 files are close.')
    parser.set_defaults(func=main)

    # define args
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-1', '--first',
                        help='1st file', required=True)
    parser.add_argument('-2', '--second',
                        help='2nd file', required=True)
    parser.add_argument('-r', '--rtol', type=float, default=1.5e-9,
                        help='rtol for numpy\'s allclose')
    parser.add_argument('-a', '--atol', type=float, default=1.5e-9,
                        help='atol for numpy\'s allclose')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='verbose to stdout.')
    parser.add_argument('-S', '--silent', action='store_true',
                        help='name to stdout when assertion failed.')

    # parsing and run main
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
