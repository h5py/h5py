# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements ObjectID base class.
"""

include "_locks.pxi"
from defs cimport *

DEF USE_LOCKING = True


# --- Locking code ------------------------------------------------------------
#
# Most of the functions and methods in h5py spend all their time in the C API
# for Python, and hold the GIL until they exit.  However, in some cases,
# particularly when calling native-Python functions from the stdlib or
# elsewhere, the GIL can be released mid-function and another thread can
# call into the API at the same time.
#
# This is bad news, especially for the object identifier registry.
#
# We serialize all access to the low-level API with a single recursive lock.
# Only one thread at a time can call any low-level routine.
#
# Note that this doesn't affect other applications like PyTables which are
# interacting with HDF5.  In the case of the identifier registry, this means
# that it's possible for an identifier to go stale and for PyTables to reuse
# it before we've had a chance to set obj.id = 0.  For this reason, h5py is
# advertised for EITHER multithreaded use OR use alongside PyTables/NetCDF4,
# but not both at the same time.

IF USE_LOCKING:
    cdef FastRLock _phil = FastRLock()
    phil = _phil

    def with_phil(func):
        """ Locking decorator """

        import functools

        def wrapper(*args, **kwds):
            with _phil:
                return func(*args, **kwds)

        functools.update_wrapper(wrapper, func, ('__name__', '__doc__'))
        return wrapper
ELSE:
    cdef BogoLock phil = BogoLock()

    def with_phil(func):
        return func

# --- End locking code --------------------------------------------------------


# --- Registry code ----------------------------------------------------------
#
# With HDF5 1.8, when an identifier is closed its value may be immediately
# re-used for a new object.  This leads to odd behavior when an ObjectID
# with an old identifier is left hanging around... for example, a GroupID
# belonging to a closed file could suddenly "mutate" into one from a
# different file.
#
# This is generally handled by setting obj.id = 0 when deallocating, or when
# explicitly closing an ObjectID instance via obj._close().  However, certain
# actions in HDF5 can *remotely* invalidate identifiers.  For example, closing
# a file opened with H5F_CLOSE_STRONG will also close open groups, etc.
#
# When such a "nonlocal" event occurs, we have to examine all live ObjectID
# instances, and manually set obj.id = 0.  That's what the function
# nonlocal_close() does.  We maintain an inventory of all live ObjectID
# instances in the registry dict.  Then, when a nonlocal event occurs,
# nonlocal_close() walks through the inventory and sets the stale identifiers
# to 0.
#
# See also __cinit__ and __dealloc__ for Object ID.

import weakref
import warnings

# Print messages to stdout with identifier diagnostic info
DEBUG_ID = False

def debug(what):
    if DEBUG_ID:
        print what

# Will map id(obj) -> weakref(obj), where obj is an ObjectID instance
cdef dict registry = {}

@with_phil
def nonlocal_close():
    """ Find dead ObjectIDs and set their integer identifiers to 0.
    """
    cdef ObjectID obj

    for python_id, ref in registry.items():

        obj = ref()
        if obj is not None:

            if (not obj.locked) and (not H5Iget_type(obj.id) > 0):
                debug("NONLOCAL - invalidating %d of kind %s HDF5 id %d" %
                        (python_id, type(obj), obj.id) )
                obj.id = 0
            else:
                debug("NONLOCAL - OK identifier %d of kind %s HDF5 id %d" %
                        (python_id, type(obj), obj.id) )

        # The ObjectID somehow died without being removed from the registry
        else:
            warnings.warn("Found murdered identifier %d of kind %s HDF5 id %d" % 
                             (python_id, type(obj), obj.id), RuntimeWarning)
            del registry[python_id]

# --- End registry code -------------------------------------------------------




cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    """

    property fileno:
        def __get__(self):
            cdef H5G_stat_t stat
            with _phil:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                return (stat.fileno[0], stat.fileno[1])

    property valid:
        def __get__(self):
            if not self.id:
                return False
            with _phil:
                return H5Iget_type(self.id) > 0

    def __cinit__(self, id_):
        self.id = id_
        self.locked = 0
        debug("CINIT - registering %d of kind %s HDF5 id %d" % (id(self), type(self), id_))
        registry[id(self)] = weakref.ref(self)

    def __dealloc__(self):
        if self.valid and (not self.locked):
            H5Idec_ref(self.id)
        debug("DEALLOC - unregistering %d of kind %s HDF5 id %d" % (id(self), type(self), self.id))
        self.id = 0
        del registry[id(self)] 

    def __nonzero__(self):
        return self.valid

    def __copy__(self):
        cdef ObjectID cpy
        cpy = type(self)(self.id)
        H5Iinc_ref(cpy.id)
        return cpy

    def __richcmp__(self, object other, int how):
        """ Default comparison mechanism for HDF5 objects (equal/not-equal)

        Default equality testing:
        1. Objects which are not both ObjectIDs are unequal
        2. Objects with the same HDF5 ID number are always equal
        3. Objects which hash the same are equal
        """
        cdef bint equal = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, ObjectID):
            if self.id == other.id:
                equal = 1
            else:
                try:
                    equal = hash(self) == hash(other)
                except TypeError:
                    pass

        if how == 2:
            return equal
        return not equal

    def _close(self):
        """ Manually close this object. """
        debug("CLOSE - %d of kind %s HDF5 id %d" % (id(self), type(self), self.id))
        if self.valid and not self.locked:
            H5Idec_ref(self.id)
        self.id = 0

    def __hash__(self):
        """ Default hashing mechanism for HDF5 objects

        Default hashing strategy:
        1. Try to hash based on the object's fileno and objno records
        2. If (1) succeeds, cache the resulting value
        3. If (1) fails, raise TypeError
        """
        cdef H5G_stat_t stat

        if self._hash is None:
            try:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                self._hash = hash((stat.fileno[0], stat.fileno[1], stat.objno[0], stat.objno[1]))
            except Exception:
                raise TypeError("Objects of class %s cannot be hashed" % self.__class__.__name__)

        return self._hash


cdef hid_t pdefault(ObjectID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    return pid.id
