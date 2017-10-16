#!/usr/bin/env python

'''
Delete HDF5 files that cannot be opened.
One reason of this happening is when the file is saved, it got interupted/terminated.

Examples:

find . -iname '*.hdf5' -exec h5delete.py -p 32 {} +
find . -path '*/coadd/*' -name '*.hdf5' -exec h5delete.py -p 32 {} +
'''
import argparse
from functools import partial

from dautil.util import get_map_parallel
from dautil.IO.h5 import h5delete

__version__ = '0.2'


def main(args):
    map_parallel = get_map_parallel(args.p)
    map_parallel(partial(h5delete, dry_run=args.dry_run, verbose=args.verbose), args.input)


def cli():
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

    main(args)

if __name__ == "__main__":
    cli()
