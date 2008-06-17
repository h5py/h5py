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
    Common module for the HDF5 low-level interface library.  

    Library version and API information lives here:
    - HDF5_VERS, HDF5_VERS_TPL:  Library version
    - API_VERS, API_VERS_TPL:  API version (1.6 or 1.8) used to compile h5py.
"""



# === API =====================================================================

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
    
#: HDF5 library version as a 3-tuple (major, minor, release), e.g. (1,6,5)
HDF5_VERS_TPL = get_libversion()        

#: HDF5 library version as a string "major.minor.release", e.g. "1.6.5"
HDF5_VERS = "%d.%d.%d" % HDF5_VERS_TPL

#: API version used to compile, as a string "major.minor", e.g. "1.6"
API_VERS = '1.6'

#: API version used to compile, as a 2-tuple (major, minor), e.g. (1,6)
API_VERS_TPL = (1,6)


class DDict(dict):
    """ Internal class.
    """
    def __missing__(self, key):
        return '*INVALID* (%s)' % str(key)

# === Identifier wrappers =====================================================

cdef class ObjectID:
        
    def __cinit__(self, hid_t id):
        """ Object init; simply records the given ID. """
        self.id = id

    def __dealloc__(self):
        """ Automatically decrefs the ID, if it's valid. """
        if H5Iget_type(self.id) != H5I_BADID:
            H5Idec_ref(self.id)

    def __copy__(self):
        """ Create a new instance and incref the ID. """
        copy = type(self)(self.id)
        H5Iinc_ref(self.id)
        return copy

    def __str__(self):
        if H5Iget_type(self.id) != H5I_BADID:
            ref = str(H5Iget_ref(self.id))
        else:
            ref = "INVALID"

        return "%d [%s] %s" % (self.id, ref, self.__class__.__name__)

cdef class LockableID(ObjectID):

    def __dealloc__(self):
        if not self._locked and H5Iget_type(self.id) != H5I_BADID:
            H5Idec_ref(self.id)

    def __copy__(self):
        return type(self)(self.id)

    def __str__(self):
        if H5Iget_type(self.id) != H5I_BADID:
            ref = str(H5Iget_ref(self.id))
        else:
            ref = "INVALID"
        if self._locked:
            lstr = "locked"
        else:
            lstr = "unlocked"

        return "%d [%s] (%s) %s" % (self.id, ref, lstr, self.__class__.__name__)


# === Library init ============================================================

cdef int import_hdf5() except -1:
    if H5open() < 0:
        raise RuntimeError("Failed to initialize the HDF5 library.")
    _enable_exceptions()
    return 0














