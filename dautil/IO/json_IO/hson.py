'''dump and load Python dict as json string in an HDF5 store
where each dict key becomes an HDF5 dataset
compression performance is not good because the compression
is not applied across datasets
'''


import h5py
import numpy as np

import ujson

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}


def dump(key, json_obj, f, compression_opts=9):
    f.create_dataset(key,
        data=np.array([ujson.dumps(json_obj).encode()]),
        compression_opts=compression_opts,
        **H5_CREATE_KW
    )


def dumps(json_dict_obj, f, compression_opts=9):
    for key, value in json_dict_obj.items():
        dump(key, value, f, compression_opts=compression_opts)


def load(key, f):
    return ujson.loads(f[key][0])
