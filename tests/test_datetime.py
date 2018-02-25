import numpy as np

from .context import datetime


def test():
    sample = '20140725_005302'
    np_time = datetime.strptime_np(sample)
    assert datetime.strptime_np_general(sample) == np_time

    assert datetime.strttime_np(np_time) == sample
    assert datetime.strttime_np_general(np_time) == sample
