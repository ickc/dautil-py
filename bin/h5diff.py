#!/usr/bin/env python

"""
For example,

find small-test -iname '*.hdf5' | sed 's/^small-test\(.*\)$/h5diff\.py -1 small-test\1 -2 small-test-reference\1 -r 1e-1 -a 1 -S/g' | xargs -i -n1 -P1 bash -c '$0' {}
find high_0_3_full -name '*.hdf5' | sed -e 's/^high_0_3_full\(.*\)$/h5diff\.py -1 high_0_3_full\1 -2 high_0_3\1 -S/g' -e 's/high_0_3\/\(.*\)null/high_0_3\/\1full/g' | xargs -i -n1 -P1 bash -c '$0' {}
"""

from __future__ import print_function

import argparse
import sys

import h5py

__version__ = "0.2"


def main(args):
    with h5py.File(args.first, "r") as f1:
        with h5py.File(args.second, "r") as f2:
            try:
                if args.datasets:
                    import numpy as np

                    for dataset in args.datasets:
                        try:
                            np.testing.assert_allclose(
                                f1[dataset], f2[dataset], args.rtol, args.atol
                            )
                        except KeyError:
                            raise AssertionError
                else:
                    from dautil.IO.h5 import h5assert_recursive

                    h5assert_recursive(
                        f1, f2, args.rtol, args.atol, verbose=args.verbose
                    )
            except AssertionError:
                if args.silent:
                    print(args.first, file=sys.stderr)
                else:
                    raise


def cli():
    parser = argparse.ArgumentParser(description="Compare if 2 HDF5 files are close.")
    parser.set_defaults(func=main)

    # define args
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    parser.add_argument("-1", "--first", help="1st file", required=True)
    parser.add_argument("-2", "--second", help="2nd file", required=True)
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        help="if specified, check only these datasets, else check resursively.",
    )
    parser.add_argument(
        "-r", "--rtol", type=float, default=1.5e-9, help="rtol for numpy's allclose"
    )
    parser.add_argument(
        "-a", "--atol", type=float, default=1.5e-9, help="atol for numpy's allclose"
    )
    parser.add_argument(
        "-V", "--verbose", action="store_true", help="verbose to stdout."
    )
    parser.add_argument(
        "-S",
        "--silent",
        action="store_true",
        help="name to stdout when assertion failed.",
    )

    # parsing and run main
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
