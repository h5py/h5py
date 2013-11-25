# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements ObjectID base class and global object registry.

    It used to be that we could store the HDF5 identifier in an ObjectID
    and simply close it when the object was deallocated.  However, since
    HDF5 1.8.5 they have started recycling object identifiers, which
    breaks this system.

    We now use a global registry of object identifiers.  This is implemented
    via a dictionary which maps an integer representation of the identifier
    to a weak reference of an ObjectID.  There is only one ObjectID instance
    in the universe for each integer identifier.  When the HDF5 reference
    count for a identifier reaches zero, HDF5 closes the object and reclaims
    the identifier. When this occurs, the identifier and weak reference must
    be deleted from the registry. If an ObjectID is deallocated, it is deleted
    from the registry and the HDF5 reference count is decreased, HDF5 closes
    and reclaims the identifier for future use.

    All interactions with the registry must be synchronized for thread safety.
    You must acquire "registry.lock" before interacting with the registry. The
    registry is not internally synchronized, in the interest of performance: we
    don't want the same thread attempting to acquire the lock multiple times
    during a single operation, if we can avoid it.

    All ObjectIDs and subclasses thereof should be opened with the "open"
    classmethod factory function, such that an existing ObjectID instance can
    be returned from the registry when appropriate.
"""

from defs cimport *

cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    """

    property fileno:
        def __get__(self):
            cdef H5G_stat_t stat
            H5Gget_objinfo(self.id, '.', 0, &stat)
            return (stat.fileno[0], stat.fileno[1])

    property valid:
        def __get__(self):
            if not self.id:
                return False
            res = H5Iget_type(self.id) > 0
            if not res:
                self.id = 0
            return res

    def __cinit__(self, id):
        self.id = id
        self.locked = 0

    def __dealloc__(self):
        try:
            H5Idec_ref(self.id)
        except Exception:
            pass

    def __nonzero__(self):
        return self.valid

    def __copy__(self):
        cdef ObjectID cpy
        cpy = type(self)(self.id)
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
        """ Close this object """
        if self.valid:
            H5Idec_ref(self.id)

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
