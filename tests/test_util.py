import numpy as np

from .context import util


def test_arange_inv():
    start, stop, step = 0., 4., 1. / 40000.
    array = np.arange(start, stop, step)
    np.testing.assert_array_equal(array, np.arange(*util.arange_inv(array)))


def test_linspace_inv():
    start, stop, num = 0., 4., 40000
    array = np.linspace(start, stop, num)
    np.testing.assert_array_equal(array, np.linspace(*util.linspace_inv(array)))


def test_running_mean():
    test = np.arange(10)
    n = 3
    assert util.running_mean_arange(0, 10, 1, 3) == (1, 9, 1)
    assert util.running_mean_linspace(0, 9, 10, 3) == (1, 8, 8)
    assert util.running_mean(test, n).size == test.size - n + 1


# def test_parallel():
#     '''somehow pytest doesn't play nice with futures
#     '''
#     def f(a, b, c):
#         return a - b, b - c, c - a

#     np.random.seed(314159)
#     temp = np.random.randint(0, 100, 9).reshape((3, 3))
#     tempT = list(map(list, zip(*temp)))
#     result = list(map(f, *temp))
#     assert result == util.map_parallel(f, *temp, processes=1)
#     assert result == util.map_parallel(f, *temp, processes=3)
#     assert result == util.map_parallel(f, *temp, processes=1, mode='multithreading')
#     assert result == util.map_parallel(f, *temp, processes=3, mode='multithreading')
#     assert result == util.map_parallel(f, *temp, processes=3)
#     assert result == util.starmap_parallel(f, tempT, processes=1)
#     assert result == util.starmap_parallel(f, tempT, processes=3)
#     assert result == util.starmap_parallel(f, tempT, processes=1, mode='multithreading')
#     assert result == util.starmap_parallel(f, tempT, processes=3, mode='multithreading')
