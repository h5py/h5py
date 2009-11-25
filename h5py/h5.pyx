# cython: profile=False

#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

"""
    Common support and versioning module for the h5py HDF5 interface.

    This is an internal module which is designed to set up the library and
    enables HDF5 exception handling.

    Exception classes are now located in the module h5py.h5e.
"""

include "config.pxi"


from h5e cimport register_thread

import atexit
import threading

IF H5PY_18API:
    ITER_INC    = H5_ITER_INC     # Increasing order
    ITER_DEC    = H5_ITER_DEC     # Decreasing order
    ITER_NATIVE = H5_ITER_NATIVE  # No particular order, whatever is fastest

    INDEX_NAME      = H5_INDEX_NAME       # Index on names      
    INDEX_CRT_ORDER = H5_INDEX_CRT_ORDER  # Index on creation order    

cdef class SmartStruct:

    """ Provides basic mechanics for structs """
    
    def _hash(self):
        raise TypeError("%s instances are unhashable" % self.__class__.__name__)

    def __hash__(self):
        # This is forwarded so that I don't have to reimplement __richcmp__ everywhere
        return self._hash()

    def __richcmp__(self, object other, int how):
        """Equality based on hash.  If unhashable, NotImplemented."""
        cdef bint truthval = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, type(self)):
            try:
                truthval = hash(self) == hash(other)
            except TypeError:
                return NotImplemented

        if how == 2:
            return truthval
        return not truthval

    def __str__(self):
        """Format "name: value" pairs recursively for public attributes"""
        mems = dict([(x, str(getattr(self, x))) for x in dir(self) if not x.startswith('_')])
        for x in mems:
            if isinstance(getattr(self,x), SmartStruct):
                mems[x] = "\n"+"\n".join(["    "+y for y in mems[x].split("\n")[1:]])
        hdr = "=== %s ===\n" % self.__class__.__name__ if self._title is None else self._title
        return hdr+"\n".join(["%s: %s" % (name, mems[name]) for name in sorted(mems)])

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
        self.API_16 = bool(H5PY_16API)
        self.API_18 = bool(H5PY_18API)
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
                r = str(val[0])
                i = str(val[1])
            except Exception:
                raise TypeError("complex_names must be a 2-tuple of strings (real, img)")
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
                f = str(val[0])
                t = str(val[1])
            except Exception:
                raise TypeError("bool_names must be a 2-tuple of strings (false, true)")
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
Diagnostic mode: %s
Complex names: %s"""

        rstr %= ("%d.%d.%d" % get_libversion(), bool(self.API_16),
                bool(self.API_18), bool(self.DEBUG),
                self.complex_names)
        return rstr

cdef H5PYConfig cfg = H5PYConfig()

cpdef H5PYConfig get_config():
    """() => H5PYConfig

    Get a reference to the global library configuration object.
    """
    return cfg

# === Public C API for object identifiers =====================================

cdef class ObjectID:

    """
        Base class for all HDF5 identifiers.

        This is an extremely thin object layer, which makes dealing with
        HDF5 identifiers a less frustrating experience.  It synchronizes
        Python object reference counts with their HDF5 counterparts, so that
        HDF5 identifiers are automatically closed when they become unreachable.

        The only (known) HDF5 property which can problematic is locked objects;
        there is no way to determine whether or not an HDF5 object is locked
        or not, without trying an operation and having it fail.  A "lock" flag
        is maintained on the Python side, and is set by methods like
        TypeID.lock(), but this is not tracked across copies.  Until HDF5
        provides something like H5Tis_locked(), this will not be fixed.

        The truth value of an ObjectID (i.e. bool(obj_id)) indicates whether
        the underlying HDF5 identifier is valid.
    """

    property _valid:
        """ Indicates whether or not this identifier points to an HDF5 object.
        """
        def __get__(self):
            return H5Iget_type(self.id) != H5I_BADID

    
    def __nonzero__(self):
        """ Truth value for object identifiers (like _valid) """
        return self._valid

    def __cinit__(self, hid_t id_):
        """ Object init; simply records the given ID. """
        self._locked = 0
        self.id = id_

    def __dealloc__(self):
        """ Automatically decrefs the ID, if it's valid. """

        if (not self._locked) and H5Iget_type(self.id) != H5I_BADID:
            H5Idec_ref(self.id)

    
    def __copy__(self):
        """ Create another object wrapper which points to the same id. 

            WARNING: Locks (i.e. datatype lock() methods) do NOT work correctly
            across copies.
        """
        cdef ObjectID copy
        copy = type(self)(self.id)
        if self._valid and not self._locked:
            H5Iinc_ref(self.id)
        copy._locked = self._locked
        return copy

    def __richcmp__(self, object other, int how):
        """ Basic comparison for HDF5 objects.  Implements only equality:

            1. Mismatched types always NOT EQUAL
            2. Try to compare object hashes
            3. If unhashable, compare identifiers
        """
        cdef bint truthval = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, type(self)):
            try:
                truthval = hash(self) == hash(other)
            except TypeError:
                truthval = self.id == other.id

        if how == 2:
            return truthval
        return not truthval

    def __hash__(self):
        """ Default hash is computed from the object header, which requires
            a file-resident object.  TypeError if this can't be done.
        """
        cdef H5G_stat_t stat

        if self._hash is None:
            try:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                self._hash = hash((stat.fileno[0], stat.fileno[1], stat.objno[0], stat.objno[1]))
            except Exception:
                raise TypeError("Objects of class %s cannot be hashed" % self.__class__.__name__)

        return self._hash

    def __repr__(self):

        ref = str(H5Iget_ref(self.id)) if self._valid else "X"
        lck = "L" if self._locked else "U"
        return "<%s [%s] (%s) %d>" % (self.__class__.__name__, ref, lck, self.id)



# === HDF5 "H5" API ===========================================================


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


def _close():
    """ Internal function; do not call unless you want to lose all your data.

    Proxy for H5close().
    """
    H5close()


def _open():
    """ Internal function; do not call unless you want to lose all your data.

    Proxy for H5open().
    """
    H5open()


# === Library init ============================================================

def _exithack():
    """ Internal function; do not call unless you want to lose all your data.
    """
    # If any identifiers have reference counts > 1 when the library closes,
    # it freaks out and dumps a message to stderr.  So we have Python dec_ref
    # everything when the interpreter's about to exit.

    cdef int count
    cdef int i
    cdef hid_t *objs

    count = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL)

    if count > 0:
        objs = <hid_t*>malloc(sizeof(hid_t)*count)
        try:
            H5Fget_obj_ids(H5F_OBJ_ALL, H5F_OBJ_ALL, count, objs)
            for i from 0<=i<count:
                while H5Iget_type(objs[i]) != H5I_BADID and H5Iget_ref(objs[i]) > 0:
                    H5Idec_ref(objs[i])
        finally:
            free(objs)

    _conv.unregister_converters()

hdf5_inited = 0

cdef int init_hdf5() except -1:
    # Initialize the library and register Python callbacks for exception
    # handling.  Safe to call more than once.
    global hdf5_inited

    import _conv
    if not hdf5_inited:
        if H5open() < 0:
            raise RuntimeError("Failed to initialize the HDF5 library.")
        register_thread()
        if register_lzf() < 0:
            raise RuntimeError("Failed to register LZF filter")
        atexit.register(_exithack)
        #h5py_register_conv()
        _conv.register_converters()
        hdf5_inited = 1

    return 0


 
# === Module init =============================================================

_hdf5_version_tuple = get_libversion()        
_api_version_tuple = (int(H5PY_API/10), H5PY_API%10)
_version_tuple = tuple([int(x) for x in H5PY_VERSION.split('-')[0].split('.')])
_version_string = H5PY_VERSION

# For backwards compatibility with user code used to the exception classes
# being in this module.
from h5e import H5Error

init_hdf5()
import _conv
