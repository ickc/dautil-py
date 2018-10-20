#!/usr/bin/env python

import argparse
from itertools import chain
from glob import iglob as glob
import os
from functools import partial

import sys
PY2 = sys.version_info[0] == 2
if PY2:
    import cPickle as pickle
else:
    import pickle

from dautil.util import map_parallel
from dautil.IO import makedirs

__version__ = '0.1'


def convert(basedir, output, in_path, protocol=2):
    out_path = in_path.replace(basedir, output, 1)
    if os.path.isfile(out_path):
        print('{} existed, skip.'.format(out_path))
        return
    makedirs(os.path.dirname(out_path))

    with open(in_path, 'rb') as f:
        data = pickle.load(f) if PY2 else pickle.load(f, encoding='latin1')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=2)
    return


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob
    in_paths = chain(*(_glob(os.path.join(args.basedir, glob_i))
                       for glob_i in args.glob))

    _convert = partial(convert, args.basedir, args.output, protocol=args.protocol)

    Nones = map_parallel(
        _convert,
        in_paths,
        mode=('mpi' if args.mpi else 'multiprocessing'),
        processes=args.processes
    )
    if args.verbose:
        print('Finish converting {} pickle files.'.format(len(Nones)))


def cli():
    parser = argparse.ArgumentParser(description='Convert pickle to pickle in a certain protocol.')

    parser.add_argument('basedir',
                        help='Base directory of input pickle files.')
    parser.add_argument('-o', '--output', required=True,
                        help='Base directory of output pickle files.')
    parser.add_argument('--glob', required=True, nargs='+',
                        help='Glob pattern from BASEDIR.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, recursive globbing, Python 3 only.')

    parser.add_argument('--protocol', type=int, default=2,
                        help='Output pickle procotol. Default: 2.')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')

    parser.add_argument('--mpi', action='store_true',
                        help='If specified, use MPI.')
    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='use p processes with multiprocessing. Hint: use total no. of threads available.')

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
