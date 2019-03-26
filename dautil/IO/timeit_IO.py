from __future__ import print_function

import timeit
import time
from functools import wraps


def timeit_IO(f):
    '''a decorator to add keyword timeit_filename to the function args
    if timeit_filename not None, dump time taken to that file
    '''
    @wraps(f)
    def f_decorated(*args, **kwargs):
        timeit_filename = kwargs.pop('timeit_filename', None)

        time = timeit.default_timer()

        result = f(*args, **kwargs)

        time -= timeit.default_timer()
        if timeit_filename is None:
            print('{},{}'.format(f, -time))
        else:
            with open(timeit_filename, 'w') as file:
                print('{},{}'.format(timeit_filename, -time), file=file)

            return result

    return f_decorated


def slowdown(f, second=1):
    @wraps(f)
    def f_decorated(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()

        time_elapsed = end - start
        print(time_elapsed)
        time.sleep(second - time_elapsed)
        return result

    return f_decorated
