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
from h5 cimport standard_richcmp
from h5p cimport propwrap, pdefault, PropFAID, PropFCID, H5P_DEFAULT
from h5t cimport typewrap
from h5a cimport AttrID
from h5d cimport DatasetID
from h5g cimport GroupID
from h5i cimport H5Iget_type, H5Iinc_ref, H5I_type_t, \
                 H5I_FILE, H5I_GROUP, H5I_ATTR, H5I_DATASET, H5I_DATATYPE
from utils cimport emalloc, efree, pybool

# Runtime imports
import h5
import h5g

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

def open(char* name, unsigned int flags=H5F_ACC_RDWR, PropFAID fapl=None):
    """ (STRING name, UINT flags=ACC_RDWR, PropFAID fapl=None)
        => FileID

        Open an existing HDF5 file.  Keyword "flags" may be ACC_RWDR or
        ACC_RDONLY.  Keyword fapl may be a file access property list.
    """
    return FileID(H5Fopen(name, flags, pdefault(fapl)))

def create(char* name, int flags=H5F_ACC_TRUNC, PropFCID fcpl=None,
                                                PropFAID fapl=None):
    """ (STRING name, INT flags=ACC_TRUNC, PropFCID fcpl=None,
                                           PropFAID fapl=None)
        => FileID

        Create a new HDF5 file.  Keyword "flags" may be either:
            ACC_TRUNC:  Truncate an existing file, discarding its data
            ACC_EXCL:   Fail if a conflicting file exists

        To keep the behavior in line with that of Python's built-in functions,
        the default is ACC_TRUNC.  Be careful!
    """
    return FileID(H5Fcreate(name, flags, pdefault(fcpl), pdefault(fapl)))

def flush(ObjectID obj not None, int scope=H5F_SCOPE_LOCAL):
    """ (ObjectID obj, INT scope=SCOPE_LOCAL)

        Tell the HDF5 library to flush file buffers to disk.  "obj" may
        be the file identifier, or the identifier of any object residing in
        the file.  Keyword "scope" may be:
            SCOPE_LOCAL:    Flush only the given file
            SCOPE_GLOBAL:   Flush the entire virtual file
    """
    H5Fflush(obj.id, <H5F_scope_t>scope)

def is_hdf5(char* name):
    """ (STRING name) => BOOL is_hdf5

        Determine if a given file is an HDF5 file.  Note this raises an 
        exception if the file doesn't exist.
    """
    return pybool(H5Fis_hdf5(name))

def mount(ObjectID loc not None, char* name, FileID fid not None):
    """ (ObjectID loc, STRING name, FileID fid)
    
        Mount an open file as "name" under group loc_id.
    """
    H5Fmount(loc.id, name, fid.id, H5P_DEFAULT)
    
def unmount(ObjectID loc not None, char* name):
    """ (ObjectID loc, STRING name)

        Unmount a file, mounted as "name" under group loc_id.
    """
    H5Funmount(loc.id, name)

def get_name(ObjectID obj not None):
    """ (ObjectID obj) => STRING file_name
        
        Determine the name of the file in which the specified object resides.
    """
    cdef ssize_t size
    cdef char* name
    name = NULL

    size = H5Fget_name(obj.id, NULL, 0)
    assert size >= 0
    name = <char*>emalloc(sizeof(char)*(size+1))
    try:    
        H5Fget_name(obj.id, name, size+1)
        pname = name
        return pname
    finally:
        efree(name)

def get_obj_count(object where=OBJ_ALL, int types=H5F_OBJ_ALL):
    """ (OBJECT where=OBJ_ALL, types=OBJ_ALL) => INT n_objs

        Get the number of open objects.

        where:  Either a FileID instance representing an HDF5 file, or the
                special constant OBJ_ALL, to count objects in all files.

        type:   Specify what kinds of object to include.  May be one of OBJ_*, 
                or any bitwise combination (e.g. OBJ_FILE | OBJ_ATTR).  

                The special value OBJ_ALL matches all object types, and 
                OBJ_LOCAL will only match objects opened through a specific 
                identifier.
    """
    cdef hid_t where_id
    if isinstance(where, FileID):
        where_id = where.id
    elif isinstance(where, int) or isinstance(where, long):
        where_id = where
    else:
        raise TypeError("Location must be a FileID or OBJ_ALL.")

    return H5Fget_obj_count(where_id, types)

cdef object wrap_identifier(hid_t ident):
    # Support function for get_obj_ids

    cdef H5I_type_t typecode
    cdef ObjectID obj
    typecode = H5Iget_type(ident)
    if typecode == H5I_FILE:
        obj = FileID(ident)
    elif typecode == H5I_DATASET:
        obj = DatasetID(ident)
    elif typecode == H5I_GROUP:
        obj = GroupID(ident)
    elif typecode == H5I_ATTR:
        obj = AttrID(ident)
    elif typecode == H5I_DATATYPE:
        obj = typewrap(ident)
    else:
        raise ValueError("Unrecognized type code %d" % typecode)

    # The HDF5 function doesn't seem to inc_ref these identifiers.
    H5Iinc_ref(ident)
    return obj

def get_obj_ids(object where=OBJ_ALL, int types=H5F_OBJ_ALL):
    """ (OBJECT where=OBJ_ALL, types=OBJ_ALL) => LIST open_ids

        Get a list of identifier instances for open objects.

        where:  Either a FileID instance representing an HDF5 file, or the
                special constant OBJ_ALL, to list objects in all files.

        type:   Specify what kinds of object to include.  May be one of OBJ_*, 
                or any bitwise combination (e.g. OBJ_FILE | OBJ_ATTR).  

                The special value OBJ_ALL matches all object types, and 
                OBJ_LOCAL will only match objects opened through a specific 
                identifier.
    """
    cdef int count
    cdef int i
    cdef hid_t where_id
    cdef hid_t *obj_list
    cdef list py_obj_list
    obj_list = NULL
    py_obj_list = []

    if isinstance(where, FileID):
        where_id = where.id
    elif isinstance(where, int) or isinstance(where, long):
        where_id = where
    else:
        raise TypeError("Location must be a FileID or OBJ_ALL.")

    try:
        count = H5Fget_obj_count(where_id, types)
        obj_list = <hid_t*>emalloc(sizeof(hid_t)*count)

        H5Fget_obj_ids(where_id, types, count, obj_list)
        for i from 0<=i<count:
            py_obj_list.append(wrap_identifier(obj_list[i]))
        return py_obj_list

    finally:
        efree(obj_list)

# === FileID implementation ===================================================

cdef class FileID(ObjectID):

    """ 
        Represents an HDF5 file identifier.

        These objects wrap a small portion of the H5F interface; all the
        H5F functions which can take arbitrary objects in addition to
        file identifiers are provided as functions in the h5f module.

        Properties:
        name:   File name on disk
    """

    property name:
        """ File name on disk (according to h5f.get_name()) """
        def __get__(self):
            return get_name(self)

    def close(self):
        """ ()

            Terminate access through this identifier.  Note that depending on
            what property list settings were used to open the file, the
            physical file might not be closed until all remaining open
            identifiers are freed.  
        """
        H5Fclose(self.id)

    def reopen(self):
        """ () => FileID

            Retrieve another identifier for a file (which must still be open).
            The new identifier is guaranteed to neither be mounted nor contain
            a mounted file.
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
        return propwrap(H5Fget_create_plist(self.id))

    def get_access_plist(self):
        """ () => PropFAID

            Retrieve a copy of the property list which manages access 
            to this file.
        """
        return propwrap(H5Fget_access_plist(self.id))

    def get_freespace(self):
        """ () => LONG freespace

            Determine the amount of free space in this file.  Note that this
            only tracks free space until the file is closed.
        """
        return H5Fget_freespace(self.id)
    
    def __richcmp__(self, object other, int how):
        return standard_richcmp(self, other, how)

    def __hash__(self):
        # Obtain the file number from the root group metadata
        if self._hash is None:
            info = h5g.get_objinfo(self)
            self._hash = hash(info.fileno)
        return self._hash




