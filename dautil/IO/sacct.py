import subprocess
import sys

import pandas as pd

PY2 = sys.version_info[0] == 2
if PY2:
    from StringIO import StringIO
else:
    from io import StringIO

ALLFORMAT = None


def _init_allformat():
    global ALLFORMAT
    ALLFORMAT = subprocess.run(['sacct', '--helpformat'], stdout=subprocess.PIPE).stdout.decode('utf-8').split()


def get_job(jobid):
    '''run ``sacct`` for the given jobid
    and return a DataFrame of that.
    '''
    if ALLFORMAT is None:
        _init_allformat()

    result = subprocess.run(
        [
            'sacct',
            '-j', jobid,
            '--parsable',
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
    if ALLFORMAT is None:
        _init_allformat()

    result = subprocess.run(
        [
            'sacct',
            '-u', username,
            '--parsable',
            '--format={}'.format(','.join(ALLFORMAT))
        ],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    with StringIO(result) as f:
        return pd.read_csv(f, sep='|')
