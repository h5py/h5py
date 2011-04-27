
"""
    Implements a mechanism by which data can be easily shared among objects
    which reside in the same HDF5 file.
"""

import collections

filedata = collections.defaultdict(dict)

def shared(fget, fset=None, fdel=None):

    def proxy_fget(self):
        sc = filedata[self.id.fileno]
        return fget(self, sc)

    if fset is not None:
        def proxy_fset(self, val):
            sc = filedata[self.id.fileno]
            return fset(self, sc, val)
    else:
        proxy_fset = None

    if fdel is not None:
        def proxy_fdel(self):
            sc = filedata[self.id.fileno]
            return fdel(self, sc)
    else:
        proxy_fdel = None

    proxy_fget.__doc__ = fget.__doc__

    return property(proxy_fget, proxy_fset, proxy_fdel)

def getval(self, key):
    return filedata[self.id.fileno]

def setval(self, key, val):
    filedata[self.id.fileno][key] = val

def wipe(self):
    try:
        del filedata[self.id.fileno]
    except KeyError:
        pass
