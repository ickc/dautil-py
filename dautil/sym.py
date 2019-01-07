import numpy as np
from sympy.ntheory import factorint


def fft_nice(n):
    '''Given integer ``n``, check if it is at the best handling
    size fot FFTW according to
    http://www.fftw.org/fftw2_doc/fftw_3.html
    '''
    factors = factorint(n)
    max_prime = max(factors)
    if max_prime <= 7:
        return True
    elif max_prime <= 13:
        if factors.get(11, 0) + factors.get(13, 0) <= 1:
            return True
    return False


def next_fft_size(n):
    '''choose the next good FFTW size >= n
    '''
    while not fft_nice(n):
        n += 1
    return n
