import os
# from tco import with_continuations


def split(path, n=1):
    '''recursively split paths into components for ``n`` times
    assert os.path.split(path) == split(path)
    '''
    path = [path]
    for __ in range(n):
        path = list(os.path.split(path[0])) + path[1:]
    # stripping first element '' when ``path`` is a relative path
    return path if path[0] else path[1:]


def split_all(path):
    '''recursively split paths into components until exhausted
    '''
    path =[path]
    # condition to end if the first element is only os.path.sep
    # also take care of the case '//'
    while len(path[0].replace(os.path.sep, '')) > 0:
        path = list(os.path.split(path[0])) + path[1:]
    # stripping first element '' when ``path`` is a relative path
    return path if path[0] else path[1:]


# @with_continuations()
# def split_tco(path, n=1, self=None):
#     '''recursively split paths into components for ``n`` times
#     using tail call optimization (this is slower than the non-tco version)
#     assert os.path.split(path) == split([path])
#     '''
#     path = list(os.path.split(path[0])) + path[1:]
#     return self(path, n - 1) if n > 1 else path


# @with_continuations()
# def split_all_tco(path, self=None):
#     '''recursively split paths into components until exhausted
#     using tail call optimization (this is slower than the non-tco version)
#     '''
#     path = list(os.path.split(path[0])) + path[1:]
#     return self(path) if len(path[0].replace(os.path.sep, '')) > 0 else (path if path[0] else path[1:])
