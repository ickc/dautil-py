#!/usr/bin/env python

'''
Delete pkl files that cannot be opened.
One reason of this happening is when the file is saved, it got interupted/terminated.
'''

from __future__ import print_function

import argparse
import os
import sys
from functools import partial
from glob import iglob as glob
from itertools import chain

from dautil.IO import read_pkl2
from dautil.util import map_parallel

__version__ = '0.2'


def pkldelete(filename, dry_run=False, verbose=False):
    try:
        read_pkl2(filename)
        if verbose:
            print(filename, 'is good.')
    except EOFError:
        if verbose:
            print(filename, 'is not good, and will be deleted.', file=sys.stderr)
        else:
            print(filename, file=sys.stderr)
        if not dry_run:
            os.remove(filename)


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob
    pkl_in_paths = chain(*(_glob(glob_i)
                           for glob_i in args.input))

    Nones = map_parallel(
        partial(pkldelete, dry_run=args.dry_run, verbose=args.verbose),
        pkl_in_paths,
        processes=args.p
    )
    if args.verbose:
        print('Finish checking {} pkl files.'.format(len(Nones)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete pickle input if it is invalid.')

    parser.add_argument('input', nargs='+',
                        help='glob pattern of pickle files.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, recursive globbing, Python 3 only.')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Dry run, do not delete.')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')
    parser.add_argument('-p', type=int, default=1,
                        help='use p processes with multiprocessing. Hint: use total no. of threads available.')

    main(parser.parse_args())
