import numpy as np

from .context import util


def test_arange_inv():
    start, stop, step = 0., 4., 1 / 40000
    array = np.arange(start, stop, step)
    inv_array = util.arange_inv(array)
    np.testing.assert_array_equal(np.arange(start, stop, step), np.arange(*inv_array))


def test_running_mean():
    test = np.arange(10)
    n = 3
    assert util.running_mean_arange(0, 10, 1, 3) == (1, 9, 1)
    assert util.running_mean(test, n).size == test.size - n + 1
