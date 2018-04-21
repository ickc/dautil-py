def get_slice(df, name, no_levels=False, no_indexslice=False):
    '''Given ``name`` of an index from the MultiIndex
    from ``df``, return the ``level`` this name is at,
    ``levels`` of all levels except this name is at,
    and an ``indexslice`` that slices all MultiIndex

    Note: ``indexslice`` is of type list. Useful to be
    reassigned at ``level`` to another slice.
    When passed to pandas.DataFrame, need to be casted to
    tuple first.
    '''
    level = df.index.names.index(name)
    n = df.index.nlevels

    if not no_levels:
        # get a list of all levels but level (of 'name')
        levels = list(range(n))
        levels.pop(level)
    else:
        levels = None

    if not no_indexslice:
        # create empty slice of length nlevels
        indexslice = [slice(None)] * n
    else:
        indexslice = None

    return level, levels, indexslice
