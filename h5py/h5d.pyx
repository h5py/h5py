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
    Provides access to the low-level HDF5 "H5D" dataset interface.
"""

# Pyrex compile-time imports
from h5s cimport H5S_ALL, H5S_UNLIMITED, H5S_SCALAR, H5S_SIMPLE, \
                    H5Sget_simple_extent_type, H5Sclose, H5Sselect_all, \
                    H5Sget_simple_extent_ndims, H5Sget_select_npoints
from numpy cimport import_array, PyArray_DATA
from utils cimport  check_numpy_read, check_numpy_write, \
                    convert_tuple, \
                    emalloc, efree

# Runtime imports
import h5
from h5 import DDict
import h5t
import h5s

import_array()

# === Public constants and data structures ====================================

COMPACT     = H5D_COMPACT
CONTIGUOUS  = H5D_CONTIGUOUS
CHUNKED     = H5D_CHUNKED

ALLOC_TIME_DEFAULT  = H5D_ALLOC_TIME_DEFAULT
ALLOC_TIME_LATE     = H5D_ALLOC_TIME_LATE
ALLOC_TIME_EARLY    = H5D_ALLOC_TIME_EARLY
ALLOC_TIME_INCR     = H5D_ALLOC_TIME_INCR

SPACE_STATUS_NOT_ALLOCATED  = H5D_SPACE_STATUS_NOT_ALLOCATED
SPACE_STATUS_PART_ALLOCATED = H5D_SPACE_STATUS_PART_ALLOCATED
SPACE_STATUS_ALLOCATED      = H5D_SPACE_STATUS_ALLOCATED

FILL_TIME_ALLOC = H5D_FILL_TIME_ALLOC
FILL_TIME_NEVER = H5D_FILL_TIME_NEVER
FILL_TIME_IFSET = H5D_FILL_TIME_IFSET

FILL_VALUE_UNDEFINED    = H5D_FILL_VALUE_UNDEFINED
FILL_VALUE_DEFAULT      = H5D_FILL_VALUE_DEFAULT
FILL_VALUE_USER_DEFINED = H5D_FILL_VALUE_USER_DEFINED

# === Basic dataset operations ================================================

def create(ObjectID loc not None, char* name, TypeID tid not None, 
            SpaceID space not None, PropDCID plist=None):
    """ (ObjectID loc, STRING name, TypeID tid, SpaceID space,
         PropDCID plist=None ) 
        => DatasetID

        Create a new dataset under an HDF5 file or group.  Keyword plist 
        may be a dataset creation property list.
    """
    cdef hid_t plist_id
    plist_id = pdefault(plist)
    return DatasetID(H5Dcreate(loc.id, name, tid.id, space.id, plist_id))

def open(ObjectID loc not None, char* name):
    """ (ObjectID loc, STRING name) => DatasetID

        Open an existing dataset attached to a group or file object, by name.
    """
    return DatasetID(H5Dopen(loc.id, name))


# === Dataset I/O =============================================================

cdef class DatasetID(ObjectID):

    """
        Represents an HDF5 dataset identifier.

        Objects of this class may be used in any HDF5 function which expects
        a dataset identifier.  Also, all H5D* functions which take a dataset
        instance as their first argument are presented as methods of this
        class.

        Properties:
        dtype:  Numpy dtype representing the dataset type
        shape:  Numpy-style shape tuple representing the dataspace
        rank:   Integer giving dataset rank
    """

    property dtype:
        """ Numpy-style dtype object representing the dataset type """
        def __get__(self):
            cdef TypeID tid
            tid = self.get_type()
            return tid.dtype

    property shape:
        """ Numpy-stype shape tuple representing the dataspace """
        def __get__(self):
            cdef SpaceID sid
            sid = self.get_space()
            return sid.get_simple_extent_dims()

    property rank:
        """ Integer giving the dataset rank (0 = scalar) """
        def __get__(self):
            cdef SpaceID sid
            sid = self.get_space()
            return sid.get_simple_extent_ndims()

    def _close(self):
        """ ()

            Terminate access through this identifier.  You shouldn't have to
            call this manually; Dataset objects are automatically destroyed
            when their Python wrappers are freed.
        """
        H5Dclose(self.id)

    def read(self, SpaceID mspace not None, SpaceID fspace not None, 
                   ndarray arr_obj not None, PropDXID plist=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj, 
             PropDXID plist=None)

            Read data from an HDF5 dataset into a Numpy array.  For maximum 
            flexibility, you can specify dataspaces for the file and the Numpy
            object. Keyword plist may be a dataset transfer property list.

            The provided Numpy array must be writable, C-contiguous, and own
            its data.  If this is not the case, ValueError will be raised and 
            the read will fail.

            It is your responsibility to ensure that the memory dataspace
            provided is compatible with the shape of the Numpy array.  Since a
            wide variety of dataspace configurations are possible, this is not
            checked.  You can easily crash Python by reading in data from too
            large a dataspace.
        """
        cdef TypeID mtype
        cdef hid_t plist_id
        plist_id = pdefault(plist)

        mtype = h5t.py_create(arr_obj.dtype)
        check_numpy_write(arr_obj, -1)

        H5Dread(self.id, mtype.id, mspace.id, fspace.id, plist_id, PyArray_DATA(arr_obj))

        
    def write(self, SpaceID mspace not None, SpaceID fspace not None, 
                    ndarray arr_obj not None, PropDXID plist=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj, 
             PropDXID plist=None)

            Write data from a Numpy array to an HDF5 dataset. Keyword plist may 
            be a dataset transfer property list.

            The provided Numpy array must be C-contiguous, and own its data.  
            If this is not the case, ValueError will be raised and the read 
            will fail.
        """
        cdef TypeID mtype
        cdef hid_t plist_id
        plist_id = pdefault(plist)

        mtype = h5t.py_create(arr_obj.dtype)
        check_numpy_read(arr_obj, -1)

        H5Dwrite(self.id, mtype.id, mspace.id, fspace.id, plist_id, PyArray_DATA(arr_obj))

    def extend(self, object shape):
        """ (TUPLE shape)

            Extend the given dataset so it's at least as big as "shape".  Note 
            that a dataset may only be extended up to the maximum dimensions of 
            its dataspace, which are fixed when the dataset is created.
        """
        cdef int rank
        cdef hid_t space_id
        cdef hsize_t* dims
        space_id = 0
        dims = NULL

        try:
            space_id = H5Dget_space(self.id)
            rank = H5Sget_simple_extent_ndims(space_id)

            require_tuple(shape, 0, rank, "shape")

            dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(shape, dims, rank)
            H5Dextend(self.id, dims)

        finally:
            efree(dims)
            if space_id:
                H5Sclose(space_id)

    def get_space(self):
        """ () => SpaceID

            Create and return a new copy of the dataspace for this dataset.
        """
        return SpaceID(H5Dget_space(self.id))

    def get_space_status(self):
        """ () => INT space_status_code

            Determine if space has been allocated for a dataset.  
            Return value is one of:
                SPACE_STATUS_NOT_ALLOCATED
                SPACE_STATUS_PART_ALLOCATED
                SPACE_STATUS_ALLOCATED 
        """
        cdef H5D_space_status_t status
        H5Dget_space_status(self.id, &status)
        return <int>status

    def get_type(self):
        """ () => TypeID

            Create and return a new copy of the datatype for this dataset.
        """
        return typewrap(H5Dget_type(self.id))

    def get_create_plist(self):
        """ () => PropDCID

            Create an return a new copy of the dataset creation property list
            used when this dataset was created.
        """
        return propwrap(H5Dget_create_plist(self.id))

    def get_offset(self):
        """ () => LONG offset

            Get the offset of this dataset in the file, in bytes.
        """
        return H5Dget_offset(self.id)

    def get_storage_size(self):
        """ () => LONG storage_size

            Determine the amount of file space required for a dataset.  Note 
            this only counts the space which has actually been allocated; it 
            may even be zero.
        """
        return H5Dget_storage_size(self.id)



