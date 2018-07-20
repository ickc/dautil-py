import pandas as pd
import subprocess

import sys
PY2 = sys.version_info[0] == 2
if PY2:
    from StringIO import StringIO
else:
    from io import StringIO

ALLFORMAT = subprocess.run(['sacct', '--helpformat'], stdout=subprocess.PIPE).stdout.decode('utf-8').split()


def get_job(jobid):
    '''run ``sacct`` for the given jobid
    and return a DataFrame of that.
    '''
    result = subprocess.run(
        [
            'sacct',
             '-j', jobid,
             '--parsable2',
             '--format={}'.format(','.join(ALLFORMAT))
        ],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    with StringIO(result) as f:
        return pd.read_csv(f, sep='|')


def get_user(username):
    '''run ``sacct`` for the given username
    and return a DataFrame of that.
    '''
    result = subprocess.run(
        [
            'sacct',
             '-u', username,
             '--parsable2',
             '--format={}'.format(','.join(ALLFORMAT))
        ],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    with StringIO(result) as f:
        return pd.read_csv(f, sep='|')
