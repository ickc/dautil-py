#!/usr/bin/env python

'''
Delete pkl files that cannot be opened.
One reason of this happening is when the file is saved, it got interupted/terminated.
'''

from __future__ import print_function

import argparse
import os
import sys

from dautil.IO import read_pkl2

__version__ = '0.1'


def main(args):
    for filename in args.input:
        try:
            read_pkl2(filename)
            if args.verbose:
                print(filename, 'is good.')
        except IOError:
            if args.verbose:
                print(filename, 'is not good, and will be deleted.', file=sys.stderr)
            else:
                print(filename, file=sys.stderr)
            if not args.dry_run:
                os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete pickle input if it cannot be opened.')

    parser.add_argument('input', nargs='+',
                        help='Input pickle files. Can be more than 1.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Dry run, do not delete.')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Print info on the input files. If not selected, only the filename of the to-be-deleted files are print.')

    main(parser.parse_args())
