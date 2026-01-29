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
from .defs cimport *

import os

# --- Locking code ------------------------------------------------------------
#
# Most of the functions and methods in h5py spend all their time in the Cython
# code for h5py, and hold the GIL until they exit.  However, in some cases,
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

### {{if OBJECTS_USE_LOCKING}}
  ### {{if FREE_THREADING}}
from threading import RLock
_phil = RLock()
  ### {{else}}
cdef FastRLock _phil = FastRLock()
  ### {{endif}}
### {{else}}
cdef BogoLock _phil = BogoLock()
### {{endif}}

# Python alias for access from other modules
phil = _phil

def with_phil(func):
    """ Locking decorator """

    import functools

    def wrapper(*args, **kwds):
        with _phil:
            return func(*args, **kwds)

    functools.update_wrapper(wrapper, func)
    return wrapper

def _phil_before_fork():
    """
    Acquire the `phil` lock before forking so no thread other
    than the current (forking) thread is holding the lock.
    """
    _phil.acquire()

def _phil_after_fork():
    """
    Release the lock after forking in both the parent and the child.
    """
    _phil.release()

# Register fork handlers to safely handle `phil` Lock in forked child processes
# in the presence of other threads which might potentially hold the lock.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_phil_before_fork,
                        after_in_child=_phil_after_fork,
                        after_in_parent=_phil_after_fork)

# --- End locking code --------------------------------------------------------


# --- Registry code ----------------------------------------------------------
#
# With HDF5 1.8, when an identifier is closed its value may be immediately
# re-used for a new object.  This leads to odd behavior when an ObjectID
# with an old identifier is left hanging around... For example, a GroupID
# belonging to a closed file could suddenly "mutate" into one from a
# different file.
#
# There are two ways such "zombie" identifiers can arise.  The first is if
# an identifier is explicitly closed via obj._close().  For this case we
# set obj.id = 0.  The second is that certain actions in HDF5 can *remotely*
# invalidate identifiers.  For example, closing a file opened with
# H5F_CLOSE_STRONG will also close open groups, etc.
#
# When such a "nonlocal" event occurs, we have to examine all live ObjectID
# instances, and manually set obj.id = 0.  That's what the function
# nonlocal_close() does.  We maintain an inventory of all live ObjectID
# instances in the registry dict.  Then, when a nonlocal event occurs,
# nonlocal_close() walks through the inventory and sets the stale identifiers
# to 0.  It must be explicitly called; currently, this happens in FileID.close()
# as well as the high-level File.close().
#
# The entire low-level API is now explicitly locked, so only one thread at at
# time is taking actions that may create or invalidate identifiers. See the
# "locking code" section above.
#
# See also __cinit__ and __dealloc__ for class ObjectID.

import weakref
import warnings

# Will map id(obj) -> weakref(obj), where obj is an ObjectID instance.
# Objects are added only via ObjectID.__cinit__, and dropped by the weakref system
cdef object registry = weakref.WeakValueDictionary()

# These 2 functions are part of a bugfix from Python 3.14. Before this,
# a race condition meant that an entry could be removed from wvd.data while
# another thread was iterating over it. We can remove these and use the
# WeakValueDict methods once 3.14 is the minimum supported version.
# https://github.com/python/cpython/issues/89967
def wvd_values(self):
    for wr in self.data.copy().values():
        obj = wr()
        if obj is not None:
            yield obj

def wvd_items(self):
    for k, wr in self.data.copy().items():
        v = wr()
        if v is not None:
            yield k, v


@with_phil
def print_reg():
    import h5py
    objs = list(wvd_values(registry))

    files = len([x for x in objs if isinstance(x, h5py.h5f.FileID)])
    groups = len([x for x in objs if isinstance(x, h5py.h5g.GroupID)])

    print("REGISTRY: %d | %d FileID | %d GroupID" % (len(objs), files, groups))


@with_phil
def nonlocal_close():
    """ Find dead ObjectIDs and set their integer identifiers to 0.
    """
    cdef ObjectID obj
    cdef list reg_ids

    for python_id, obj in wvd_items(registry):
        # Locked objects are immortal, as they generally are provided by
        # the HDF5 library itself (property list classes, etc.).
        if obj.locked:
            continue

        # Invalid object; set obj.id = 0 so it doesn't become a zombie
        if not H5Iis_valid(obj.id):
            ### {{if OBJECTS_DEBUG_ID}}
            print("NONLOCAL - invalidating %d of kind %s HDF5 id %d" %
                  (python_id, type(obj), obj.id)
            )
            ### {{endif}}
            obj.id = 0
            continue

# --- End registry code -------------------------------------------------------


cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    attributes:
    cdef object __weakref__
    cdef readonly hid_t id
    cdef public int locked              # Cannot be closed, explicitly or auto
    cdef object _hash
    cdef size_t _pyid
    """

    @property
    def fileno(self):
        cdef H5G_stat_t stat
        with _phil:
            H5Gget_objinfo(self.id, '.', 0, &stat)
            return (stat.fileno[0], stat.fileno[1])


    @property
    def valid(self):
        return is_h5py_obj_valid(self)


    def __cinit__(self, id_):
        with _phil:
            self.id = id_
            self.locked = 0
            self._pyid = id(self)
            ### {{if OBJECTS_DEBUG_ID}}
            print("CINIT - registering %d of kind %s HDF5 id %d" % (self._pyid, type(self), self.id))
            ### {{endif}}
            registry[self._pyid] = self


    def __dealloc__(self):
        self._dealloc()

    # During interpreter shutdown, module attributes are set to None
    # before __dealloc__ and __del__ methods are executed.
    def _dealloc(self, _phil=_phil, warn=warnings.warn):
        with _phil:
            ### {{if OBJECTS_DEBUG_ID}}
            print("DEALLOC - unregistering %d HDF5 id %d" % (self._pyid, self.id))
            ### {{endif}}
            if is_h5py_obj_valid(self) and (not self.locked):
                if H5Idec_ref(self.id) < 0:
                    warn(
                        "Reference counting issue with HDF5 id {}".format(
                            self.id
                        )
                    )

    def _close(self):
        """ Manually close this object. """

        with _phil:
            ### {{if OBJECTS_DEBUG_ID}}
            print("CLOSE - %d HDF5 id %d" % (self._pyid, self.id))
            ### {{endif}}
            if is_h5py_obj_valid(self) and (not self.locked):
                if H5Idec_ref(self.id) < 0:
                    warnings.warn(
                        "Reference counting issue with HDF5 id {}".format(
                            self.id
                        )
                    )
            self.id = 0

    def close(self):
        """ Close this identifier. """
        # Note this is the default close method.  Subclasses, e.g. FileID,
        # which have nonlocal effects should override this.
        self._close()

    def __bool__(self):
        return self.valid

    def __copy__(self):
        cdef ObjectID cpy
        with _phil:
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

        with _phil:
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


    def __hash__(self):
        """ Default hashing mechanism for HDF5 objects

        Default hashing strategy:
        1. Try to hash based on the object's fileno and objno records
        2. If (1) succeeds, cache the resulting value
        3. If (1) fails, raise TypeError
        """
        cdef H5G_stat_t stat

        with _phil:
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


# Note: _phil=_phil allows this to work during interpreter shutdown.
# Read note at ObjectID.__dealloc__.
cdef int is_h5py_obj_valid(ObjectID obj, _phil=_phil):
    """
    Check that h5py object is valid, i.e. HDF5 object wrapper is valid and HDF5
    object is valid
    """
    # MUST BE CALLABLE AT ANY TIME, CANNOT USE PROPERTIES ETC. AS PER
    # http://cython.readthedocs.io/en/latest/src/userguide/special_methods.html

    # Locked objects are always valid, regardless of obj.id
    if obj.locked:
        return True

    # Former zombie object
    if obj.id == 0:
        return False

    # Ask HDF5.  Note that H5Iis_valid only works for "user"
    # identifiers, hence the above checks.
    with _phil:
        return H5Iis_valid(obj.id)
