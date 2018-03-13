import numpy as np

from .context import date_time


def test():
    sample = '20140725_005302'
    np_time = date_time.strptime_np(sample)
    assert date_time.strptime_np_general(sample) == np_time

    assert date_time.strttime_np(np_time) == sample
    assert date_time.strttime_np_general(np_time) == sample
