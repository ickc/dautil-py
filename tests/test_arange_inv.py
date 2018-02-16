import numpy as np

from .context import util

def test_arange_inv():
    start, stop, step = 0., 4., 1 / 40000
    array = np.arange(start, stop, step)
    inv_array = util.arange_inv(array)
    np.testing.assert_array_equal(np.arange(start, stop, step), np.arange(*inv_array))
