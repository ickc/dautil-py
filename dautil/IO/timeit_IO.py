from __future__ import print_function

import timeit
from functools import wraps


def timeit_IO(f):
    '''a decorator to add keyword timeit_filename to the function args
    if timeit_filename not None, dump time taken to that file
    '''
    @wraps(f)
    def f_decorated(*args, **kwargs):
        timeit_filename = kwargs.pop('timeit_filename', None)

        if timeit_filename is None:
            return f(*args, **kwargs)
        else:
            time = timeit.default_timer()

            result = f(*args, **kwargs)

            time -= timeit.default_timer()
            with open(timeit_filename, 'w') as file:
                print('{},{}'.format(timeit_filename, -time), file=file)

            return result

    return f_decorated
