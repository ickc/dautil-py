#!/usr/bin/env python

import argparse
from functools import partial

import h5py

from dautil.util import get_map_parallel

__version__ = 0.1


def del_dataset(dataset, h5path):
    with h5py.File(h5path, "w") as f:
        for i in dataset:
            try:
                del f[i]
            except KeyError:
                pass


def main(args):
    map_parallel(partial(del_dataset, args.dataset), args.input)


def cli():
    parser = argparse.ArgumentParser(
        description="Delete datasets from input HDF5 files."
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="*",
        required=True,
        help="a list of path to HDF5 files where datasets are to be deleted from.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        nargs="*",
        required=True,
        help="a list of the names of datasets to be deleted.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=1,
        help="No. of parallel processes using multiprocessing.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    args = parser.parse_args()

    global map_parallel
    map_parallel = get_map_parallel(args.p)

    main(args)


if __name__ == "__main__":
    cli()
