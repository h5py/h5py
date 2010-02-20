
# Cython has limits on what you can declare inside control structures.  This
# native-Python module is a shim to allow things like dynamic class
# definitions and functional closures.

def generate_class(cls1, cls2):
    """ Create a new class from two bases.  The new name is the concatenation
    of cls2.__name__ with "H5"; e.g. KeyError -> KeyErrorH5.
    """
    class HybridClass(cls1, cls2):
        pass
    HybridClass.__name__ = cls2.__name__
    return HybridClass

def _exc(func):
    """ Enable native HDF5 exception handling on this function """
    import functools
    from h5e import capture_errors, release_errors

    def _exception_proxy(*args, **kwds):
        capture_errors()
        try:
            return func(*args, **kwds)
        finally:
            release_errors()

    functools.update_wrapper(_exception_proxy, func)
    return _exception_proxy


