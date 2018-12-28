#!/usr/bin/env python

'''Concat pandas DataFrame
'''

from __future__ import print_function

import argparse
from itertools import chain
from glob import iglob as glob
from functools import partial

import pandas as pd

from dautil.util import map_parallel

__version__ = '0.1'


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob
    h5_in_paths = chain(*(_glob(glob_i)
                          for glob_i in args.input))

    dfs = map_parallel(
        pd.read_hdf,
        h5_in_paths,
        processes=args.p
    )
    print('Loaded {} input, concat...'.format(len(dfs)))
    df = pd.concat(dfs)
    del h5_in_paths, dfs
    df.sort_index(inplace=True)
    df.to_hdf(
        args.output,
        'df',
        format='table',
        complevel=args.compress_level,
        fletcher32=True
    )


def cli():
    parser = argparse.ArgumentParser(description='Concat DataFrames stored in HDF5 given glob patterns.')

    parser.add_argument('input', nargs='+',
                        help='glob pattern of HDF5 files.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output HDF5 path to store the DataFrame.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, recursive globbing, Python 3 only.')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument('-p', type=int, default=1,
                        help='use p processes with multiprocessing. Hint: use total no. of threads available.')
    parser.add_argument('-c', '--compress-level', default=9, type=int,
                        help='compress level of gzip algorithm. Default: 9.')

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
