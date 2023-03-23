import base64
import logging
import pickle
import zlib

def native_object_decoded(s):
    s = base64.b64decode(s)
    s = zlib.decompress(s)
    s = pickle.loads(s)
    return s


def native_object_encoded(x):
    x = pickle.dumps(x)
    x = zlib.compress(x)
    x = base64.b64encode(x).decode()
    return x


def uncompress(s):
    s = zlib.decompress(s)
    s = pickle.loads(s)
    return s


def compress(x):
    x = pickle.dumps(x)
    x = zlib.compress(x)
    return x