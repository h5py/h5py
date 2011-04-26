include "config.pxi"

from defs cimport *

from _errors import H5Error # for backwards compatibility

ITER_INC    = H5_ITER_INC     # Increasing order
ITER_DEC    = H5_ITER_DEC     # Decreasing order
ITER_NATIVE = H5_ITER_NATIVE  # No particular order, whatever is fastest

INDEX_NAME      = H5_INDEX_NAME       # Index on names      
INDEX_CRT_ORDER = H5_INDEX_CRT_ORDER  # Index on creation order    

cdef class H5PYConfig:

    """
        Provides runtime access to global library settings.  You retrieve the
        master copy of this object by calling h5py.get_config().

        API_16 (T/F, readonly)
            Is the HDF5 1.6 API available?  Currently always true.

        API_18 (T/F, readonly)
            If the HDF5 1.8 API available?

        complex_names (tuple, r/w)
            Settable 2-tuple controlling how complex numbers are saved.
            Defaults to ('r','i').

        bool_names (tuple, r/w)
            Settable 2-tuple controlling the HDF5 enum names used for boolean
            values.  Defaults to ('FALSE', 'TRUE') for values 0 and 1.
    """

    def __init__(self):
        self.API_16 = False
        self.API_18 = True
        self._r_name = 'r'
        self._i_name = 'i'
        self._f_name = 'FALSE'
        self._t_name = 'TRUE'

    property complex_names:
        """ Settable 2-tuple controlling how complex numbers are saved.

        Format is (real_name, imag_name), defaulting to ('r','i').
        """

        def __get__(self):
            return (self._r_name, self._i_name)

        def __set__(self, val):
            try:
                if len(val) != 2: raise TypeError()
                r = str(val[0])
                i = str(val[1])
            except Exception:
                raise TypeError("complex_names must be a length-2 sequence of names (real, img)")
            self._r_name = r
            self._i_name = i

    property bool_names:
        """ Settable 2-tuple controlling HDF5 ENUM names for boolean types.

        Format is (false_name, real_name), defaulting to ('FALSE', 'TRUE').
        """
        def __get__(self):
            return (self._f_name, self._t_name)

        def __set__(self, val):
            try:
                if len(val) != 2: raise TypeError()
                f = str(val[0])
                t = str(val[1])
            except Exception:
                raise TypeError("bool_names must be a length-2 sequence of of names (false, true)")
            self._f_name = f
            self._t_name = t

    def __repr__(self):
        rstr =  \
"""\
Summary of h5py config
======================
HDF5: %s
1.6 API: %s
1.8 API: %s
Complex names: %s"""

        rstr %= ("%d.%d.%d" % get_libversion(), bool(self.API_16),
                bool(self.API_18), self.complex_names)
        return rstr

cdef H5PYConfig cfg = H5PYConfig()

cpdef H5PYConfig get_config():
    """() => H5PYConfig

    Get a reference to the global library configuration object.
    """
    return cfg

def get_libversion():
    """ () => TUPLE (major, minor, release)

        Retrieve the HDF5 library version as a 3-tuple.
    """
    cdef unsigned int major
    cdef unsigned int minor
    cdef unsigned int release
    cdef herr_t retval
    
    H5get_libversion(&major, &minor, &release)

    return (major, minor, release)





