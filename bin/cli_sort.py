#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys
from functools import partial

from dautil.parser import cli_sort

__version__ = "0.1"


def main(args):
    for line in args.infile:
        print((cli_sort(line, args.n)), file=args.output)


def cli():
    parser = argparse.ArgumentParser(description='Sort cli args per line of input.')

    parser.add_argument('infile', nargs='?',
                        type=argparse.FileType('r'),
                        help='input text file.', default=sys.stdin)
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        help='output text file.', default=sys.stdout)
    parser.add_argument('-n', type=int,
                        help='number of postional argument including the command.', default=1)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    main(parser.parse_args())


if __name__ == "__main__":
    cli()
