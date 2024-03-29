#!/usr/bin/env python

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

__version__ = '0.1.1'

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


def main(args):
    pathnames = np.array([
        pathname
        for pathname in Path(args.dir).glob('**/*')
        if (pathname.is_file() or pathname.is_symlink()) and pathname.suffix in EXTS
    ])
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        durations = np.array(list(executor.map(get_duration, pathnames)), dtype=np.float64)
    pd.DataFrame(
        {
            'Path': pathnames,
            'duration': durations
        }
    ).to_hdf(args.output, 'df', mode='w', complevel=9)


def cli():
    parser = argparse.ArgumentParser(description="Get video duration within a directory.")

    parser.add_argument('dir',
                        help='Directories.')
    parser.add_argument('-o', '--output',
                        help='Output pandas hdf5 file name.')
    parser.add_argument('-p', '--processes', type=int,
                        help='No. of parallel processes.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(cli())
