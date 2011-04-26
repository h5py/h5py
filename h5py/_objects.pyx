
"""
    Implements ObjectID base class and global object registry.

    It used to be that we could store the HDF5 identifier in an ObjectID
    and simply close it when the object was deallocated.  However, since
    HDF5 1.8.5 they have started recycling object identifiers, which
    breaks this system.

    We now use a global registry of "proxy objects".  This is implemented
    via a weak-value dictionary which maps an integer representation of the
    identifier to an IDProxy object.  There is only one IDProxy object in
    the universe for each integer identifier.  Objects hold strong references
    to this "master" IDProxy object.  As they are deallocated, the
    reference count of the IDProxy decreases.  The use of a weak-value
    dictionary means that as soon as no ObjectIDs remain which reference the
    IDProxy, it is deallocated, closing the HDF5 integer identifier in
    its __dealloc__ method.
"""

from defs cimport *

import weakref
registry = weakref.WeakValueDictionary()

cdef class IDProxy:

    property valid:
        def __get__(self):
            return H5Iget_type(self.id) > 0

    def __cinit__(self, id):
        self.id = id
        self.locked = 0

    def __dealloc__(self):
        if self.id > 0 and (not self.locked) and H5Iget_type(self.id) > 0:
            H5Idec_ref(self.id)

cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    """

    property id:
        def __get__(self):
            return self.proxy.id
        def __set__(self, id):
            cdef IDProxy newproxy = IDProxy(id)
            self.proxy = registry.setdefault(id, newproxy)
            if newproxy is not self.proxy:
                newproxy.id = 0

    property locked:
        def __get__(self):
            return self.proxy.locked
        def __set__(self, val):
            self.proxy.locked = val

    def __cinit__(self, id):
        self.id = id

    def __nonzero__(self):
        return self.proxy.valid

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


