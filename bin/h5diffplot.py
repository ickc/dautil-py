#!/usr/bin/env python

'''
For example,

find smallest-test -path '*/coadd/*' -name '*.hdf5' | sed 's/^smallest-test\(.*\)$/h5diffplot\.py -1 smallest-test\1 -2 smallest-test-no-intermediate\1 -o "\$(dirname smallest-test-no-intermediate\1)"/g' | parallel
'''

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

__version__ = '0.1'


def plot_diff_hdf5(f1, f2, out_dir, prefix='', verbose=False):
    if isinstance(f1, h5py._hl.dataset.Dataset):
        name = '-'.join([prefix] + f1.name.split('/')[1:])
        if verbose:
            print('{} is dataset, plotting...'.format(name))
        temp1 = np.nan_to_num(f1)
        temp2 = np.nan_to_num(f2)
        temp3 = np.abs(temp2 - temp1)
        # plt.plot(temp1)
        # plt.plot(temp2)
        plt.plot(temp3)
        plt.savefig(os.path.join(out_dir, name + '.png'))
        plt.close()
    elif isinstance(f1, h5py._hl.group.Group):
        if verbose:
            name = '-'.join([prefix] + f1.name.split('/')[1:])
            print('{} is group, entering...'.format(name))
        for i in f1:
            plot_diff_hdf5(f1[i], f2[i], out_dir, prefix, verbose)


def main(args):
    with h5py.File(args.first, "r") as f1:
        with h5py.File(args.second, "r") as f2:
            plot_diff_hdf5(f1, f2, args.o, prefix=os.path.splitext(os.path.basename(args.first))[0], verbose=args.verbose)


def cli():
    parser = argparse.ArgumentParser(description='Plot diff. of 2 HDF5 files.')
    parser.set_defaults(func=main)

    # define args
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    parser.add_argument('-1', '--first',
                        help='1st file', required=True)
    parser.add_argument('-2', '--second',
                        help='2nd file', required=True)
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='verbose to stdout.')
    parser.add_argument('-o',
                        help='output directory.')

    # parsing and run main
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
