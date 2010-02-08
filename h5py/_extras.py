import posixpath as pp
import math

def cproperty(attrname):
    """ Cached property using instance dict. """
    import functools
    def outer(meth):
        def inner(self):
            if attrname in self.__dict__:
                return self.__dict__[attrname]
            return self.__dict__.setdefault(attrname, meth(self))
        functools.update_wrapper(inner, meth)
        return property(inner)
    return outer

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
