import posixpath as pp
import math

def basename(name):
    """ Basename function with more readable handling of trailing slashes"""
    name = pp.basename(pp.normpath(name))
    return name if name != '' else '/'

def sizestring(size):
    """ Friendly representation of byte sizes """
    d = int(math.log(size, 1024) // 1) if size else 0
    suffix = {1: 'k', 2: 'M', 3: 'G', 4: 'T'}.get(d)
    if suffix is None:
        return "%d bytes" % size
    return "%.1f%s" % (size / (1024.**d), suffix)
