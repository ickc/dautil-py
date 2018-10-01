#!/usr/bin/env python

import argparse
from itertools import chain
from collections import deque
from glob import iglob as glob
import numpy as np
import os
import re
from word2number import w2n

__version__ = 0.1


def sort_paths(paths, patterns):
    '''``paths``: list of strings
    ``pattern``: iterables of patterns to match stem (filename w/o extension)
    per path. The matched pattern (1st if multiple mathced),
    assuming to be English numerals, will be converted to numbers. And the
    input iterables is sorted according and returned as list.
    '''
    paths = np.asarray(paths)
    match_numerals = [re.compile(pattern) for pattern in patterns]
    nums = np.array([
        w2n.word_to_num(next(chain(
            *(match_numeral.findall(os.path.splitext(os.path.basename(path))[0])
            for match_numeral in match_numerals)
        )))
        for path in paths
    ])
    idxs = nums.argsort()
    return paths[idxs]


def main(args):
    _glob = partial(glob, recursive=True) if args.recursive else glob
    paths = chain(*(_glob(path) for path in args.path))
    paths = sort_paths(list(paths), args.pattern)
    deque(map(print, paths), 0)


def cli():
    parser = argparse.ArgumentParser(description="Find and sort filenames containing numerals.")

    parser.add_argument('path', nargs='+',
                        help='glob patterns of files/directories.')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='If specified, recursive globbing, Python 3 only.')
    parser.add_argument('-p', '--pattern', nargs='+',
                        help="Regex patterns.")
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(cli())
