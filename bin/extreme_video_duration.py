#!/usr/bin/env python

import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
import sys
from moviepy.editor import VideoFileClip

from dautil.stat import get_cutoffs

__version__ = 0.1

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


def main_per_path(path):
    path = Path(path)
    files = pd.DataFrame(
        [
            file
            for file in path.glob('**/*')
            if (file.is_file() or file.is_symlink()) and file.suffix in EXTS
        ],
        columns=('Path',)
    )
    files['duration'] = files['Path'].map(lambda file: VideoFileClip(str(file)).duration)
    cutoffs = get_cutoffs(files.duration)
    return files.loc[(files.duration < cutoffs[0]) | (files.duration > cutoffs[1]), 'Path']


def main(args):
    for path in args.path:
        for file in main_per_path(path):
            print(file, file=args.output)


def cli():
    parser = argparse.ArgumentParser(description="Detect extreme video duration within a directory.")

    parser.add_argument('path', nargs='+',
                        help='glob patterns of files/directories.')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(cli())
