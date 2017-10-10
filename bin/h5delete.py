#!/usr/bin/env python

'''
Delete HDF5 files that cannot be opened.
One reason of this happening is when the file is saved, it got interupted/terminated.

Examples:

find . -iname '*.hdf5' -exec h5delete.py -p 32 {} +
find . -path '*/coadd/*' -name '*.hdf5' -exec h5delete.py -p 32 {} +
'''

from __future__ import print_function

import argparse
import h5py
import sys
import os
from functools import partial

__version__ = '0.2'


# will be overridden if args.p is > 1
map_parallel = map


def h5delete(dry_run, verbose, filename):
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

def main(args):
    map_parallel(partial(h5delete, args.dry_run, args.verbose), args.input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete HDF5 input if it cannot be opened.')

    parser.add_argument('input', nargs='+',
                        help='Input HDF5 files. Can be more than 1.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Dry run, do not delete.')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')
    parser.add_argument('-p', type=int, default=1,
                        help='use p processes with multiprocessing. Hint: use total no. of threads available.')

    args = parser.parse_args()

    # use multiprocessing if args.p > 1
    if args.p > 1:
        import multiprocessing
        pool = multiprocessing.Pool(processes=args.p)
        global map_parallel
        map_parallel = pool.map
    elif args.p < 1:
        import sys
        print('-p cannot be smaller than 1.', file=sys.stderr)
        exit(1)

    main(args)
