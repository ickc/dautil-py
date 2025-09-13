#!/usr/bin/env python

import argparse
import sys

import pyotp

__version__ = 0.1


def main(token_file):
    token = token_file.read()
    totp = pyotp.TOTP(token)
    print(totp.now())


def cli():
    parser = argparse.ArgumentParser(description="Generate TOTP from token.")

    parser.add_argument(
        "token_file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    args = parser.parse_args()

    main(args.token_file)


if __name__ == "__main__":
    cli()
