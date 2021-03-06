#!/usr/bin/env python

'''
Detect invalid HDF5 files and delete them.
'''

from __future__ import print_function

import argparse
from functools import partial
from glob import iglob as glob
from itertools import chain

from dautil.IO.h5 import h5delete
from dautil.util import map_parallel

__version__ = '0.3'


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob
    h5_in_paths = chain(*(_glob(glob_i)
                          for glob_i in args.input))

    Nones = map_parallel(
        partial(h5delete, datasets=args.datasets, dry_run=args.dry_run, verbose=args.verbose),
        h5_in_paths,
        processes=args.p
    )
    if args.verbose:
        if args.datasets:
            print('Finish checking {} HDF5 files with datasets {}.'.format(len(Nones), ' '.join(args.datasets)))
        else:
            print('Finish checking {} HDF5 files.'.format(len(Nones)))


def cli():
    parser = argparse.ArgumentParser(description='Delete HDF5 input if it is invalid.')

    parser.add_argument('input', nargs='+',
                        help='glob pattern of HDF5 files.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, recursive globbing, Python 3 only.')

    parser.add_argument('-d', '--datasets', nargs='*',
                        help='if specified, also check these datasets exists in the HDF5 file.')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Dry run, do not delete.')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')
    parser.add_argument('-p', type=int, default=1,
                        help='use p processes with multiprocessing. Hint: use total no. of threads available.')

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
