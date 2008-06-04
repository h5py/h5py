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
from defs_c cimport malloc, free
from h5  cimport herr_t, hid_t, htri_t, hssize_t
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

SCOPE_LOCAL     = H5F_SCOPE_LOCAL
SCOPE_GLOBAL    = H5F_SCOPE_GLOBAL

CLOSE_WEAK = H5F_CLOSE_WEAK
CLOSE_SEMI = H5F_CLOSE_SEMI
CLOSE_STRONG = H5F_CLOSE_STRONG
CLOSE_DEFAULT = H5F_CLOSE_DEFAULT

OBJ_FILE    = H5F_OBJ_FILE
OBJ_DATASET = H5F_OBJ_DATASET
OBJ_GROUP   = H5F_OBJ_GROUP
OBJ_DATATYPE = H5F_OBJ_DATATYPE
OBJ_ATTR    = H5F_OBJ_ATTR
OBJ_ALL     = H5F_OBJ_ALL
OBJ_LOCAL   = H5F_OBJ_LOCAL

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

def reopen(hid_t file_id):
    """ (INT file_id) => INT new_file_id

        Retrieve another identifier for a file (which must still be open).
    """
    cdef hid_t retval
    retval = H5Freopen(file_id)
    if retval < 0:
        raise FileError("Faile to re-open file %d" % file_id)
    return retval

def mount(hid_t loc_id, char* name, hid_t file_id, hid_t plist_id=H5P_DEFAULT):
    """ (INT loc_id, STRING name, INT file_id, INT, plist_id=H5P_DEFAULT)
    
        Mount an open file as "name" under group loc_id.  If present, plist_id 
        is a mount property list.
    """
    cdef herr_t retval
    retval = H5Fmount(loc_id, name, file_id, plist_id)
    if retval < 0:
        raise FileError('Failed to mount file %s as "%s" under group %d' % (file_id, name, loc_id))
    
def unmount(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Unmount a file, mounted as "name" under group loc_id.
    """
    cdef herr_t retval
    retval = H5Funmount(loc_id, name)
    if retval < 0:
        raise FileError('Failed to unmount "%s" under group %d' % (name, loc_id))

# === File inspection =========================================================

def get_filesize(hid_t file_id):
    """ (INT file_id) => LONG size_in_bytes

        Determine the total size of the HDF5 file, including the user block.
    """
    cdef herr_t retval
    cdef hsize_t size
    retval = H5Fget_filesize(file_id, &size)
    if retval < 0:
        raise FileError("Can't determine size of file %d" % file_id)
    return size

def get_create_plist(hid_t file_id):
    """ (INT file_id) => INT plist_id

        Retrieve a copy of the property list used to create this file.
    """
    cdef hid_t retval
    retval = H5Fget_create_plist(file_id)
    if retval < 0:
        raise FileError("Can't retrieve creation property list for file %d" % file_id)
    return retval

def get_access_plist(hid_t file_id):
    """ (INT file_id) => INT plist_id

        Retrieve a copy of the property list which manages access to this file.
    """
    cdef hid_t retval
    retval = H5Fget_access_plist(file_id)
    if retval < 0:
        raise FileError("Can't retrieve access property list for file %d" % file_id)
    return retval

def get_freespace(hid_t file_id):
    """ (INT file_id) => LONG free space

        Determine the amount of free space in this file.  Note that this only
        tracks free space until the file is closed.
    """
    cdef hssize_t retval
    retval = H5Fget_freespace(file_id)
    if retval < 0:
        raise FileError("Can't determine free space in file %d" % file_id)
    return retval

def get_name(hid_t obj_id):
    """ (INT obj_id) => STRING file_name
        
        Determine the name of the file in which the specified object resides.
    """
    cdef ssize_t retval
    cdef char* name
    name = NULL

    retval = H5Fget_name(obj_id, NULL, 0)
    if retval < 0:
        raise FileError("Can't determine file name associated with object %d" % obj_id)

    name = <char*>malloc(retval+1)
    retval = H5Fget_name(obj_id, name, retval+1)
    if retval < 0:
        free(name)
        raise FileError("Can't determine file name associated with object %d" % obj_id)

    pname = name
    free(name)
    return pname
    
def get_obj_count(hid_t file_id, int types):
    """ (INT file_id, INT types) => INT n_objs

        Get the number of open objects in the file.  The value of "types" may
        be one of h5f.OBJ_*, or any bitwise combination (e.g. 
        OBJ_FILE | OBJ_ATTR).  The special value OBJ_ALL matches all object
        types, and OBJ_LOCAL will only match objects opened through this
        specific identifier.
    """
    cdef int retval
    retval = H5Fget_obj_count(file_id, types)
    if retval < 0:
        raise FileError("Can't determine number of open identifiers of types %d in file %d" % (types, file_id))
    return retval

def get_obj_ids(hid_t file_id, int types):
    """ (INT file_id, INT types) => LIST open_ids

        Get a list of identifiers for open objects in the file.  The value of 
        "types" may be one of h5f.OBJ_*, or any bitwise combination (e.g. 
        OBJ_FILE | OBJ_ATTR).  The special value OBJ_ALL matches all object
        types, and OBJ_LOCAL will only match objects opened through this
        specific identifier.

    """
    cdef int count
    cdef int retval
    cdef hid_t *obj_list
    cdef int i

    obj_list = NULL
    py_obj_list = []

    try:
        count = H5Fget_obj_count(file_id, types)
        if count < 0:
            raise FileError("Failed to count open identifiers.")

        obj_list = <hid_t*>malloc(sizeof(hid_t)*count)
        retval = H5Fget_obj_ids(file_id, types, count, obj_list)
        if retval < 0:
            raise FileError("Failed to enumerate open objects of types %d in file %d" % (types, file_id))

        for i from 0<=i<count:
            py_obj_list.append(obj_list[i])
    finally:
        free(obj_list)

    return py_obj_list

# === Python extensions =======================================================

PY_SCOPE = DDict({  H5F_SCOPE_LOCAL: 'LOCAL SCOPE', 
                    H5F_SCOPE_GLOBAL: 'GLOBAL SCOPE' })
PY_CLOSE = DDict({ H5F_CLOSE_WEAK: 'CLOSE WEAK', 
                    H5F_CLOSE_SEMI: 'CLOSE SEMI', 
                    H5F_CLOSE_STRONG: 'CLOSE STRONG', 
                    H5F_CLOSE_DEFAULT: 'DEFAULT CLOSE STRENGTH' })
PY_OBJ = DDict({ H5F_OBJ_FILE: 'FILE', H5F_OBJ_DATASET: 'DATASET',
                H5F_OBJ_GROUP: 'GROUP', H5F_OBJ_DATATYPE: 'DATATYPE',
                H5F_OBJ_ATTR: 'ATTRIBUTE', H5F_OBJ_ALL: 'ALL', 
                H5F_OBJ_LOCAL: 'LOCAL' })
PY_ACC = DDict({ H5F_ACC_TRUNC: 'TRUNCATE', H5F_ACC_EXCL: 'EXCLUSIVE ACCESS',
                 H5F_ACC_RDWR: 'READ-WRITE', H5F_ACC_RDONLY: 'READ-ONLY' })








    
