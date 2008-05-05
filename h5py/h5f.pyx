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
    Low-level operations on HDF5 file objects.

    Functions in this module raise errors.FileError.
"""

# Pyrex compile-time imports
from h5  cimport herr_t, hid_t, htri_t
from h5p cimport H5P_DEFAULT

# Runtime imports
import h5
from h5 import DDict
from errors import FileError

# === Public constants and data structures ====================================

ACC_TRUNC   = H5F_ACC_TRUNC
ACC_EXCL    = H5F_ACC_EXCL
ACC_RDWR    = H5F_ACC_RDWR
ACC_RDONLY  = H5F_ACC_RDONLY
ACC_MAPPER  = {H5F_ACC_TRUNC: 'TRUNCATE', H5F_ACC_EXCL: 'EXCLUSIVE',
               H5F_ACC_RDWR: 'READ-WRITE', H5F_ACC_RDONLY: 'READ-ONLY' }
ACC_MAPPER  = DDict(ACC_MAPPER)

SCOPE_LOCAL     = H5F_SCOPE_LOCAL
SCOPE_GLOBAL    = H5F_SCOPE_GLOBAL
SCOPE_MAPPER    = {H5F_SCOPE_LOCAL: 'LOCAL SCOPE', H5F_SCOPE_GLOBAL: 'GLOBAL SCOPE'}
SCOPE_MAPPER    = DDict(SCOPE_MAPPER)

CLOSE_WEAK = H5F_CLOSE_WEAK
CLOSE_SEMI = H5F_CLOSE_SEMI
CLOSE_STRONG = H5F_CLOSE_STRONG
CLOSE_DEFAULT = H5F_CLOSE_DEFAULT
CLOSE_MAPPER = {H5F_CLOSE_WEAK: 'WEAK', H5F_CLOSE_SEMI: 'SEMI', 
                H5F_CLOSE_STRONG: 'STRONG', H5F_CLOSE_DEFAULT: 'DEFAULT'}
CLOSE_MAPER = DDict(CLOSE_MAPPER)

# === File operations =========================================================

def open(char* name, unsigned int flags=H5F_ACC_RDWR, hid_t access_id=H5P_DEFAULT):
    """ (STRING name, UINT flags=ACC_RDWR, INT access_id=H5P_DEFAULT)
        => INT file_id

        Open an existing HDF5 file.  Keyword "flags" may be ACC_RWDR or
        ACC_RDONLY.  Keyword "access_id" may be a file access property list.
    """
    cdef hid_t retval
    retval = H5Fopen(name, flags, access_id)

    if retval < 0:
        raise FileError("Failed to open file '%s'" % name)
    return retval

def close(hid_t file_id):
    """ (INT file_id)
    """
    cdef herr_t retval
    retval = H5Fclose(file_id)
    if retval < 0:
        raise FileError("Failed to close file id %d" % file_id)

def create(char* name, int flags=H5F_ACC_TRUNC, hid_t create_id=H5P_DEFAULT, hid_t access_id=H5P_DEFAULT):
    """ (STRING name, INT flags=ACC_TRUNC, INT create_id=H5P_DEFAULT,
            INT access_id=H5P_DEFAULT)
        => INT file_id

        Create a new HDF5 file.  Keyword "flags" may be either ACC_TRUNC, in
        which case any existing file will be destroyed, or ACC_EXCL, which
        will force the creation to fail if the file already exists.
        Keywords create_id and access_id may be dataset creation and access
        property lists, respectively.
    """
    cdef hid_t retval
    retval = H5Fcreate(name, flags, create_id, access_id)

    if retval < 0:
        raise FileError('Failed to create file "%s" mode %d' % (name,flags))
    return retval

def flush(hid_t file_id, int scope=H5F_SCOPE_LOCAL):
    """ (INT file_id, INT scope=SCOPE_LOCAL)

        Tell the HDF5 library to flush file buffers to disk.  See the HDF5
        docs for the meaning of the scope keyword.
    """
    cdef herr_t retval
    retval = H5Fflush(file_id, <H5F_scope_t>scope)

    if retval < 0:
        raise FileError("Failed to flush file %d" % file_id)

def is_hdf5(char* name):
    """ (STRING name) => BOOL is_hdf5

        Determine if a given file is an HDF5 file.  Note this raises an 
        exception if the file doesn't exist.
    """
    cdef htri_t retval
    retval = H5Fis_hdf5(name)
    if retval < 0:
        raise FileError("Can't determine status of file '%s'" % name)
    return bool(retval)

    











    
