'''dump and load Python dict as json string in an tar ball
where each dict key becomes a file with that name (i.e. dict key
has to be a valid filename)
This allow loading a value from a key by seeking,
(instead of loading everything as a dict)
however seeking is still quite slow because there's
no index.

File object example:

with tarfile.open('tson.json.tar.gz', 'w:gz') as f:
    dumps(data, f)
'''

import tarfile
from io import BytesIO

import ujson


def dump(key, json_obj, f):
    string = ujson.dumps(json_obj).encode()
    info = tarfile.TarInfo(name=key)
    info.size = len(string)
    f.addfile(info, BytesIO(string))


def dumps(json_dict_obj, f):
    for key, value in json_dict_obj.items():
        dump(key, value, f)


def load(key, f):
    return ujson.loads(f.extractfile(key).read())
