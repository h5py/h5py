from defs cimport *

ITER_INC    = H5_ITER_INC     # Increasing order
ITER_DEC    = H5_ITER_DEC     # Decreasing order
ITER_NATIVE = H5_ITER_NATIVE  # No particular order, whatever is fastest

INDEX_NAME      = H5_INDEX_NAME       # Index on names      
INDEX_CRT_ORDER = H5_INDEX_CRT_ORDER  # Index on creation order    

class ByteStringContext(object):

    def __init__(self):
        self._readbytes = False

    def __bool__(self):
        return self._readbytes

    def __nonzero__(self):
        return self.__bool__()

    def __enter__(self):
        self._readbytes = True

    def __exit__(self, *args):
        self._readbytes = False

cdef class H5PYConfig:

    """
        Provides runtime access to global library settings.  You retrieve the
        master copy of this object by calling h5py.get_config().

        complex_names (tuple, r/w)
            Settable 2-tuple controlling how complex numbers are saved.
            Defaults to ('r','i').

        bool_names (tuple, r/w)
            Settable 2-tuple controlling the HDF5 enum names used for boolean
            values.  Defaults to ('FALSE', 'TRUE') for values 0 and 1.
    """

    def __init__(self):
        self._r_name = b'r'
        self._i_name = b'i'
        self._f_name = b'FALSE'
        self._t_name = b'TRUE'
        self._bytestrings = ByteStringContext()

    property complex_names:
        """ Settable 2-tuple controlling how complex numbers are saved.

        Format is (real_name, imag_name), defaulting to ('r','i').
        """

        def __get__(self):
            import sys
            def handle_val(val):
                if sys.version[0] == '3':
                    return val.decode('utf8')
                return val
            return (handle_val(self._r_name), handle_val(self._i_name))

        def __set__(self, val):
            def handle_val(val):
                if isinstance(val, unicode):
                    return val.encode('utf8')
                elif isinstance(val, bytes):
                    return val
                else:
                    return bytes(val)
            try:
                if len(val) != 2: raise TypeError()
                r = handle_val(val[0])
                i = handle_val(val[1])
            except Exception:
                raise TypeError("complex_names must be a length-2 sequence of strings (real, img)")
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

    property read_byte_strings:
        """ Returns a context manager which forces all strings to be returned
        as byte strings. """
        
        def __get__(self):
            return self._bytestrings

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





