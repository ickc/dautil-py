import h5py
import os
import sys

def h5delete(filename, dry_run=True, verbose=False):
    try:
        with h5py.File(filename, "r") as f:
            if verbose:
                print(filename, 'is good.')
    except IOError:
        if verbose:
            print(filename, 'is not good, and will be deleted.', file=sys.stderr)
        else:
            print(filename, file=sys.stderr)
        if not dry_run:
            os.remove(filename)
