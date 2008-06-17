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
"""

# Pyrex compile-time imports
from h5p cimport PropID, pdefault, H5P_DEFAULT
from utils cimport emalloc, efree, pybool

# Runtime imports
import h5
from h5 import DDict

# === Public constants and data structures ====================================

ACC_TRUNC   = H5F_ACC_TRUNC
ACC_EXCL    = H5F_ACC_EXCL
ACC_RDWR    = H5F_ACC_RDWR
ACC_RDONLY  = H5F_ACC_RDONLY

SCOPE_LOCAL     = H5F_SCOPE_LOCAL
SCOPE_GLOBAL    = H5F_SCOPE_GLOBAL

CLOSE_WEAK  = H5F_CLOSE_WEAK
CLOSE_SEMI  = H5F_CLOSE_SEMI
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

def open(char* name, unsigned int flags=H5F_ACC_RDWR, ):
    """ (STRING name, UINT flags=ACC_RDWR, PropID accesslist=None)
        => FileID

        Open an existing HDF5 file.  Keyword "flags" may be ACC_RWDR or
        ACC_RDONLY.
    """
    cdef hid_t plist_id
    plist_id = pdefault(accesslist)
    return FileID(H5Fopen(name, flags, accesslist))

def close(FileID file_id):
    """ (FileID file_id)
    """
    H5Fclose(file_id.id)

def create(char* name, int flags=H5F_ACC_TRUNC, PropID createlist=None,
                                                PropID accesslist=None):
    """ (STRING name, INT flags=ACC_TRUNC, PropID createlist=None,
                                           PropID accesslist=None)
        => FileID

        Create a new HDF5 file.  Keyword "flags" may be either:
            ACC_TRUNC:  Truncate an existing file, discarding its data
            ACC_EXCL:   Fail if a conflicting file exists

        To keep the behavior in line with that of Python's built-in functions,
        the default is ACC_TRUNC.  Be careful!
    """
    cdef hid_t create_id
    cdef hid_t access_id
    create_id = pdefault(createlist)
    access_id = pdefault(accesslist)
    return FileID(H5Fcreate(name, flags, create_id, access_id))

def is_hdf5(char* name):
    """ (STRING name) => BOOL is_hdf5

        Determine if a given file is an HDF5 file.  Note this raises an 
        exception if the file doesn't exist.
    """
    return pybool(H5Fis_hdf5(name))

def mount(ObjectID loc_id not None, char* name, FileID file_id not None, 
          PropID mountlist=None):
    """ (ObjectID loc_id, STRING name, FileID file_id, PropID mountlist=None)
    
        Mount an open file as "name" under group loc_id.  If present, mountlist
        is a mount property list.
    """
    cdef hid_t plist_id
    plist_id = pdefault(mountlist)
    H5Fmount(loc_id.id, name, file_id.id, plist_id)
    
def unmount(ObjectID loc_id not None, char* name):
    """ (ObjectID loc_id, STRING name)

        Unmount a file, mounted as "name" under group loc_id.
    """
    H5Funmount(loc_id.id, name)

def get_name(ObjectID obj_id not None):
    """ (INT obj_id) => STRING file_name
        
        Determine the name of the file in which the specified object resides.
    """
    cdef ssize_t size
    cdef char* name
    name = NULL

    size = H5Fget_name(obj_id.id, NULL, 0)
    assert size >= 0
    name = <char*>emalloc(sizeof(char)*(size+1))
    try:    
        H5Fget_name(obj_id.id, name, size+1)
        pname = name
        return pname
    finally:
        efree(name)

# === XXXX ===

cdef class FileID(ObjectID):

    """ 
        Represents an HDF5 file identifier.
    """

    def flush(self, int scope=H5F_SCOPE_LOCAL):
        """ (INT scope=SCOPE_LOCAL)

            Tell the HDF5 library to flush file buffers to disk.  file_id may
            be the file identifier, or the identifier of any object residing in
            the file.  Keyword "scope" may be:
                SCOPE_LOCAL:    Flush only the given file
                SCOPE_GLOBAL:   Flush the entire virtual file
        """
        H5Fflush(self.id, <H5F_scope_t>scope)


    def reopen(self):
        """ () => INT new_file_id

            Retrieve another identifier for a file (which must still be open).
        """
        return FileID(H5Freopen(self.id))


    def get_filesize(self):
        """ () => LONG size

            Determine the total size (in bytes) of the HDF5 file, 
            including any user block.
        """
        cdef hsize_t size
        H5Fget_filesize(self.id, &size)
        return size

    def get_create_plist(self):
        """ () => PropFCID

            Retrieve a copy of the property list used to create this file.
        """
        return PropFCID(H5Fget_create_plist(self.id))

    def get_access_plist(self):
        """ () => PropFAID

            Retrieve a copy of the property list which manages access 
            to this file.
        """
        return PropFAID(H5Fget_access_plist(self.id))

    def get_freespace(self):
        """ () => LONG free space

            Determine the amount of free space in this file.  Note that this only
            tracks free space until the file is closed.
        """
        return H5Fget_freespace(self.id)


    
    def get_obj_count(self, int types=H5F_OBJ_ALL):
        """ (INT types=OBJ_ALL) => INT n_objs

            Get the number of open objects in the file.  The value of "types" 
            may be one of h5f.OBJ_*, or any bitwise combination (e.g. 
            OBJ_FILE | OBJ_ATTR).  The special value OBJ_ALL matches all object
            types, and OBJ_LOCAL will only match objects opened through this
            specific identifier.
        """
        return H5Fget_obj_count(self.id, types)

    def get_obj_ids(self, int types=H5F_OBJ_ALL):
        """ (INT types=OBJ_ALL) => LIST open_ids

            Get a list of identifiers for open objects in the file.  The value of 
            "types" may be one of h5f.OBJ_*, or any bitwise combination (e.g. 
            OBJ_FILE | OBJ_ATTR).  The special value OBJ_ALL matches all object
            types, and OBJ_LOCAL will only match objects opened through this
            specific identifier.
        """
        cdef int count
        cdef hid_t *obj_list
        cdef int i
        obj_list = NULL

        py_obj_list = []
        try:
            count = H5Fget_obj_count(self.id, types)
            obj_list = <hid_t*>emalloc(sizeof(hid_t)*count)

            H5Fget_obj_ids(self.id, types, count, obj_list)
            for i from 0<=i<count:
                py_obj_list.append(obj_list[i])
            return py_obj_list

        finally:
            efree(obj_list)





    
