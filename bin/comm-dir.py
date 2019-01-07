#!/usr/bin/env python

import argparse
import os
from functools import partial
from glob import iglob as glob

from dautil.util import compose

__version__ = '0.1'


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob

    filter_funcs = []
    if args.basename:
        filter_funcs.append(os.path.basename)
    if args.remove_extension:
        filter_funcs.append(lambda x: os.path.splitext(x)[0])

    in_files = set((
        compose(*filter_funcs)(file)
        for file in _glob(os.path.join(args.indir, args.in_glob))
    ))
    out_files = set((
        compose(*filter_funcs)(file)
        for file in _glob(os.path.join(args.outdir, args.out_glob))
    ))

    diff = in_files - out_files
    if args.verbose:
        print('There are {} input files and {} output files.'.format(len(in_files), len(out_files)))
        if len(diff) > 0:
            print('There are {} files that are in input directory but not in output directory.'.format(len(diff)))
            print("And here's it:")
    print('\n'.join(diff))


def cli():
    parser = argparse.ArgumentParser(description='Compare 2 directories and print differences.')

    parser.add_argument('indir',
                        help='Input directory.')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory.')
    parser.add_argument('-I', '--in-glob', default='*',
                        help='Input glob pattern. Default: *')
    parser.add_argument('-O', '--out-glob', default='*',
                        help='Output glob pattern. Default: *')
    parser.add_argument('-b', '--basename', action='store_true',
                        help='If specified, only output basename of files.')
    parser.add_argument('-e', '--remove-extension', action='store_true',
                        help='If specified, remove the file extensions.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, glob recursively.')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
