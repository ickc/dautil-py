from .context import path_util


def test_split():
    test_path = '/global/cscratch1/sd/largepatch_null/high_0_6_coadd/null_realmaps/20140725_014555/coadd/realmap_AMP_2F_BY_BOLO_0.hdf5'
    assert path_util.split(test_path, 3) == [
        '/global/cscratch1/sd/largepatch_null/high_0_6_coadd/null_realmaps',
        '20140725_014555',
        'coadd',
        'realmap_AMP_2F_BY_BOLO_0.hdf5'
    ]
    assert path_util.split_all(test_path) == ['/',
                                              'global',
                                              'cscratch1',
                                              'sd',
                                              'largepatch_null',
                                              'high_0_6_coadd',
                                              'null_realmaps',
                                              '20140725_014555',
                                              'coadd',
                                              'realmap_AMP_2F_BY_BOLO_0.hdf5'
                                              ]
