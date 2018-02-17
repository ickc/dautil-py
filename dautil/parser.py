import numpy as np
from collections import OrderedDict
import os
from functools import partial
import pandas as pd
import sys


def cli_sort(cli, n_command):
    '''assume cli is a command with args
    where ``n_command`` is not number of positional arguments in the beginning
    including the command itself
    return a sorted cli.
    It might be useful to normalize 2 commands for taking diff.

    # TODO
    Limitations:

    - assume space always is the seperator of command/args/options
        - what if more than single space? space in options, etc.
    '''
    # get command and args
    arg_list = cli.split()
    command = arg_list[:n_command]
    arg_list = arg_list[n_command:]

    # get all starting index
    mask = np.array([i[0] == '-' for i in arg_list])
    index = mask * np.arange(len(arg_list))
    del mask
    # remember 0 is always a starting index but not included below
    index = index[index != 0]
    # compensate for index 0
    n_arg = index.shape[0] + 1
    # prepending -1 is just for convenience
    index = np.concatenate(([0], index, [-1]))
    # compensate for last index being -1
    arg_list.append('')

    # joining the options per index to one single arg
    arg_list = [' '.join(arg_list[index[i]:index[i + 1]]) for i in range(n_arg)]
    del index, n_arg
    arg_list.sort()
    return ' '.join(command + arg_list)

# GNU time output


def parse_gnu_time_line(text):
    '''Parse a line of GNU time's output
    Returns a list of 2 elements as the key-value pair
    '''
    # when there's error in running the script,
    # the first line does not start with '\t'.
    if text[0] != '\t':
        raise ValueError
    # get key-value pair
    result = text.strip().split(': ')

    # casting value to an appropriate type
    try:
        result[1] = float(result[1])
        if result[0] == 'Maximum resident set size (kbytes)':
            result = ['Maximum resident set size (GiB)', result[1] / 1048576.]
    # only command, percentage, timedelta syntax is not floatable
    except ValueError:
        # percentage
        try:
            result[1] = float(result[1][:-1]) / 100.
        except ValueError:
            # timedelta
            try:
                # TODO: not using pandas?
                if result[1].count(':') == 1:
                    result[1] = '0:' + result[1]
                result[1] = pd.to_timedelta(result[1]).total_seconds()
                # the only row with timedelta syntax is this one
                result[0] = 'Elapsed (wall clock) time (seconds)'
            # command string
            except ValueError:
                result[1] = result[1][1:-1]
    return result


def parse_gnu_time_file(filename, isdatetime=False):
    '''Parse the output of GNU time
    filename: a path to the GNU time output
    Returns a DataFrame
    '''
    # use filename as index
    index = os.path.splitext(os.path.basename(filename))[0]
    if isdatetime:
        try:
            index = pd.to_datetime(index, format='%Y%m%d_%H%M%S')
        except ValueError:
            pass

    with open(filename, 'r') as f:
        try:
            return pd.DataFrame(OrderedDict(map(parse_gnu_time_line, f)), index=(index,))
        except ValueError:
            print(filename, file=sys.stderr)
            raise ValueError


def parse_gnu_time_files(filenames, isdatetime=False, map_parallel=map):
    '''Parse the outputs of GNU time
    filenames: an iterator of paths to GNU time's outputs
    Returns a DataFrame
    '''
    df = pd.concat(map_parallel(partial(parse_gnu_time_file, isdatetime=isdatetime), filenames))
    df.sort_index(inplace=True)
    return df
