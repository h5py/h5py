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
    Private initialization module for the h5* family of modules.

    Common module for the HDF5 low-level interface library.  This module
    is imported at the top of every h5* sub-module.  Initializes the
    library and defines common version info, classes and functions.

    Library version and API information lives here:
    - HDF5_VERS, HDF5_VERS_TPL:  Library version
    - API_VERS, API_VERS_TPL:  API version (1.6 or 1.8) used to compile
"""
from h5t cimport H5Tset_overflow
from errors import H5LibraryError

import h5e

# === Library init ============================================================

_hdf5_imported = False
def import_hdf5():
    global _hdf5_imported
    if not _hdf5_imported:
        H5open()
        h5e._enable_exceptions()
        _hdf5_imported = False

# === API =====================================================================

def get_libversion():
    """ () => TUPLE (major, minor, release)

        Retrieve the HDF5 library version as a 3-tuple.
    """
    cdef unsigned int major
    cdef unsigned int minor
    cdef unsigned int release
    cdef herr_t retval
    
    retval = H5get_libversion(&major, &minor, &release)
    if retval < 0:
        raise H5LibraryError("Error determining HDF5 library version.")

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

