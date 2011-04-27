
"""
    Implements a mechanism by which data can be easily shared among objects
    which reside in the same HDF5 file.
"""


import collections

class SharedConfig():
    pass

filedata = collections.defaultdict(SharedConfig)

def shared(obj):
    return filedata[obj.id.fileno]

def wipe(obj):
    try:
        del filedata[obj.id.fileno]
    except KeyError:
        pass
