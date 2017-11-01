import numpy as np

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
