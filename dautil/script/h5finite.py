#!/usr/bin/env python

'''Assert HDF5 input is non-zero.
Print to stderr if not.
For example,

find . -iname '*.hdf5' -exec h5zero.py {} +
'''

from __future__ import print_function

import argparse
import sys
from functools import partial
from glob import iglob as glob
from itertools import chain

import h5py

from dautil.IO.h5 import h5assert_isfinite

__version__ = '0.1'


def _h5assert_isfinite(filename, verbose=False):
    with h5py.File(filename, "r") as f:
        try:
            h5assert_isfinite(f, verbose=verbose)
        except AssertionError:
            print('File {} has non-finite elements.'.format(filename), file=sys.stderr)


def main(args):
    in_paths = chain(*(glob(glob_i) for glob_i in args.input))

    __h5assert_isfinite = partial(_h5assert_isfinite, verbose=args.verbose)
    if args.use_mpi:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            executor.map(__h5assert_isfinite, in_paths)
    elif args.p > 1:
        from dautil.util import get_map_parallel
        map_parallel = get_map_parallel(args.p)
        map_parallel(__h5assert_isfinite, in_paths)
    else:
        list(map(__h5assert_isfinite, in_paths))


def cli():
    parser = argparse.ArgumentParser(description='Assert HDF5 input is finite.')
    parser.set_defaults(func=main)

    # define args
    parser.add_argument('input', nargs='+',
                        help='glob pattern. Can be more than 1.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-p', type=int, default=1,
                        help="No. of parallel processes using multiprocessing.")
    parser.add_argument('--use-mpi', action='store_true',
                        help='If specified, use MPI.')

    # parsing and run main
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
