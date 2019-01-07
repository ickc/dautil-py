#!/usr/common/software/python/3.6-anaconda-4.4/bin/python

'''Pretty print output from ``scontrol show burst``
'''

import os
import subprocess
from collections import OrderedDict

import pandas as pd

AB = '  Allocated Buffers:'
PUBU = '  Per User Buffer Use:'


def scontrol_show_burst_parser(cat_list):
    '''Expect input format is each section from ``scontrol show burst``
    e.g. the sections are either the content of
    '  Allocated Buffers:' or '  Per User Buffer Use:'
    return a DataFrame of the content
    '''
    return pd.concat(
        (
            pd.DataFrame(
                OrderedDict(item.split('=') for item in line.strip().split()),
                index=(i,)
            )
            for i, line in enumerate(cat_list)
        )
    )


def parse_size(df, col):
    '''Assume column at ``col`` is sizes ended with MiB/GiB/TiB
    and convert them in TiB
    '''
    def MiB_to_TiB(x):
        try:
            size = int(x[:-3])
            unit = x[-3:]
        except (IndexError, ValueError):
            if x in ('', '0'):
                size = 0
                unit = None
            else:
                print(x, 'cannot be parsed.')
                raise ValueError

        if unit == 'MiB':
            return size / 1048576.
        elif unit == 'GiB':
            return size / 1024.
        elif unit == 'TiB' or unit is None:
            return float(size)
        else:
            print(x, 'cannot be parsed.')
            raise ValueError

    df[col + ' (TiB)'] = df[col].apply(MiB_to_TiB)
    df.drop(col, 1, inplace=True)


def main():
    '''run ``scontrol show burst`` and return allocated_buffers, per_user_buffer_use as DataFrames
    '''
    burst_list = subprocess.run(['scontrol', 'show', 'burst'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

    i = burst_list.index(AB)
    j = burst_list.index(PUBU)

    allocated_buffers = scontrol_show_burst_parser(burst_list[i + 1:j - 1])
    per_user_buffer_use = scontrol_show_burst_parser(burst_list[j + 1:-1])
    del burst_list, i, j

    parse_size(per_user_buffer_use, 'Used')

    parse_size(allocated_buffers, 'Size')
    allocated_buffers['CreateTime'] = pd.to_datetime(allocated_buffers['CreateTime'])
    allocated_buffers.set_index('CreateTime', drop=True, inplace=True)

    return allocated_buffers, per_user_buffer_use


if __name__ == "__main__":
    allocated_buffers, per_user_buffer_use = main()
    pd.set_option('display.width', os.get_terminal_size().columns)
    print(AB, allocated_buffers, PUBU, per_user_buffer_use, sep='\n\n')
