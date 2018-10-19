#!/usr/bin/env python

import argparse
from collections import deque
import numpy as np
from pathlib import Path
from functools import partial
import scipy
import sys
from concurrent.futures import ProcessPoolExecutor
from moviepy.editor import VideoFileClip

from dautil.stat import get_cutoffs

__version__ = '0.2'

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


def main_per_path(directory, file=sys.stdout):
    pathnames = np.array([
        pathname
        for pathname in Path(directory).glob('**/*')
        if (pathname.is_file() or pathname.is_symlink()) and pathname.suffix in EXTS
    ])
    with ProcessPoolExecutor() as executor:
        durations = np.array(list(executor.map(get_duration, pathnames)), dtype=np.float32)
    idx_nan = np.isnan(durations)
    idx_nan
    data = durations[~idx_nan]
    cutoffs = get_cutoffs(data) if data.size else np.array((np.NINF, np.inf))
    for pathname in (pathnames[idx_nan | (durations < cutoffs[0]) | (durations > cutoffs[1])]):
        print(pathname, file=file)


def main(args):
    deque(map(partial(main_per_path, file=args.output), args.dirs), 0)


def cli():
    parser = argparse.ArgumentParser(description="Detect extreme video duration within a directory.")

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
