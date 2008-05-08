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
from h5e cimport H5Eset_auto, H5E_walk_t, H5Ewalk, H5E_error_t, \
                      H5E_WALK_DOWNWARD

from errors import H5LibraryError

# Activate the library
H5open()

# Disable automatic error printing to stderr
H5Eset_auto(NULL, NULL)

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

# === Error functions =========================================================

cdef herr_t walk_cb(int n, H5E_error_t *err_desc, data):

    cdef object hstring
    hstring = err_desc.desc
    if len(hstring) == 0:
        hstring = "Error"
    else:
        hstring = '"'+hstring.capitalize()+'"'

    data.append("    "+str(n)+": "+hstring+" at "+err_desc.func_name)

    return 0

def get_error_string():
    """ Internal function; don't use directly.

        Get the HDF5 error stack contents as a string.
    """
    elist = []

    H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, elist)

    if len(elist) == 0:
        return ""
    return "HDF5 error stack:\n" + '\n'.join(elist)

