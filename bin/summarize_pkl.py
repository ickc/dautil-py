#!/usr/bin/env python

import argparse
import pickle
from pprint import pprint

from dautil.util import summarize

__version__ = "0.1"


def main(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    pprint(summarize(data))


def cli():
    parser = argparse.ArgumentParser(
        description="Summarize the content of a pickle file"
    )

    parser.add_argument("path")

    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    args = parser.parse_args()

    main(args.path)


if __name__ == "__main__":
    cli()
