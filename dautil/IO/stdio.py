import os
import sys
from functools import wraps


def redirect_stdout_stderr(f):
    '''a decorator to add keyword stdout, stderr to the function args
    if stdout, stderr is not None, redirect the stdout, stderr to
    the respective files.
    '''
    @wraps(f)
    def f_decorated(*args, **kwargs):
        try:
            stdout = kwargs.pop('stdout', None)
            if stdout:
                stdout_original = sys.stdout
                f_out = os.open(stdout, os.O_WRONLY | os.O_CREAT, 0o644)
                os.dup2(f_out, sys.stdout.fileno())

            stderr = kwargs.pop('stderr', None)
            if stderr:
                stderr_original = sys.stderr
                f_err = os.open(stderr, os.O_WRONLY | os.O_CREAT, 0o644)
                os.dup2(f_err, sys.stderr.fileno())

            return f(*args, **kwargs)
        finally:
            if stdout:
                sys.stdout.flush()  # prevent buffer not saved
                os.dup2(stdout_original.fileno(), f_out)
                os.close(f_out)

            if stderr:
                sys.stderr.flush()  # prevent buffer not saved
                os.dup2(stderr_original.fileno(), f_err)
                os.close(f_err)

    return f_decorated
