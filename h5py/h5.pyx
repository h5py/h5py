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

    ** Not for public use. **

    Common module for the HDF5 low-level interface library.  This module
    is imported at the top of every h5* sub-module.  Initializes the
    library and defines common version info, classes and functions.
"""

from h5e cimport H5Eset_auto, H5E_walk_t, H5Ewalk, H5E_error_t, \
                      H5E_WALK_DOWNWARD

# Activate the library
H5open()

# Disable automatic error printing to stderr
H5Eset_auto(NULL, NULL)

def _getversionastuple():

    cdef unsigned int major
    cdef unsigned int minor
    cdef unsigned int release
    cdef herr_t retval
    
    retval = H5get_libversion(&major, &minor, &release)
    if retval < 0:
        raise RuntimeError("Error determining HDF5 library version")

    return (major, minor, release)
    
hdf5version = _getversionastuple()

def cycle():
    """ ()

        Force the HDF5 library to close all open objects and files, and re-
        initialize the library.
    """
    cdef herr_t retval
    H5close()
    retval = H5open()
    if retval < 0:
        raise RuntimeError("Failed to re-initialize the HDF5 library")


class DDict(dict):
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

    elist = []

    H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, elist)

    if len(elist) == 0:
        return ""
    return "HDF5 error stack:\n" + '\n'.join(elist)

