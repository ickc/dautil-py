#!/usr/bin/env python

import argparse
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import scipy
from moviepy.editor import VideoFileClip

from dautil.stat import get_cutoffs

__version__ = '0.3'

EXTS = {
    '.mp4',
    '.rmvb',
    '.mkv',
    '.ts',
    '.avi',
    '.m2ts',
    '.wmv',
    '.mpg',
}


def get_duration(pathname):
    try:
        return VideoFileClip(str(pathname)).duration
    except OSError:
        return np.nan
    except UnicodeDecodeError:
        return np.inf


def main_per_path(directory, file=sys.stdout):
    pathnames = np.array([
        pathname
        for pathname in Path(directory).glob('**/*')
        if (pathname.is_file() or pathname.is_symlink()) and pathname.suffix in EXTS
    ])
    with ProcessPoolExecutor() as executor:
        durations = np.array(list(executor.map(get_duration, pathnames)), dtype=np.float32)
    idx_ok = np.isfinite(durations)
    data = durations[idx_ok]
    cutoffs = get_cutoffs(data) if data.size > 1 else np.array((np.NINF, np.inf))

    # following 4 cases are mutually exclusive
    for pathname in (pathnames[np.isnan(durations)]):
        print('O', pathname, file=file)
    for pathname in (pathnames[np.isposinf(durations)]):
        print('U', pathname, file=file)
    for pathname in (pathnames[durations < cutoffs[0]]):
        print('S', pathname, file=file)
    # because durations can be +inf we need to exclude that
    for pathname in (pathnames[idx_ok & (durations > cutoffs[1])]):
        print('L', pathname, file=file)


def main(args):
    deque(map(partial(main_per_path, file=args.output), args.dirs), 0)


def cli():
    parser = argparse.ArgumentParser(description="""Detect extreme video duration within a directory.
First 2 characters are flags:
'O' means OSError when reading the file;
'U' means UnicodeDecodeError when reading the file;
'S' means duration seems too short;
'L' means duration seems too long.
"""
                                     )

    parser.add_argument('dirs', nargs='+',
                        help='Directories.')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(cli())
