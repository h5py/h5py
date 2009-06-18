
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

