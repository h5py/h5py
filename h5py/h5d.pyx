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
__doc__ = \
"""
    Provides access to the low-level HDF5 "H5D" dataset interface.
"""
include "config.pxi"

# Compile-time imports
from h5 cimport init_hdf5
from numpy cimport ndarray, import_array, PyArray_DATA, NPY_WRITEABLE
from utils cimport  check_numpy_read, check_numpy_write, \
                    convert_tuple, emalloc, efree
from h5t cimport TypeID, typewrap, py_create
from h5s cimport SpaceID
from h5p cimport PropID, propwrap, pdefault

# Initialization
import_array()
init_hdf5()

# Runtime imports
from _sync import sync, nosync

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


# === Dataset operations ======================================================

@sync
def create(ObjectID loc not None, char* name, TypeID tid not None, 
            SpaceID space not None, PropID dcpl=None):
    """ (ObjectID loc, STRING name, TypeID tid, SpaceID space,
         PropDCID dcpl=None ) 
        => DatasetID

        Create a new dataset under an HDF5 file or group.  Keyword dcpl
        may be a dataset creation property list.
    """
    return DatasetID(H5Dcreate(loc.id, name, tid.id, space.id, pdefault(dcpl)))

@sync
def open(ObjectID loc not None, char* name):
    """ (ObjectID loc, STRING name) => DatasetID

        Open an existing dataset attached to a group or file object, by name.
    """
    return DatasetID(H5Dopen(loc.id, name))

# --- Proxy functions for safe(r) threading -----------------------------------

# It's not legal to call PyErr_Occurred() with nogil, so we can't use
# the standard except * syntax.  Trap negative return numbers and convert them
# to something Cython can recognize.

cdef int H5PY_H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf) nogil except -1:

    cdef herr_t retval
    retval = H5Dread(dset_id, mem_type_id,mem_space_id, file_space_id,
                        plist_id, buf)
    if retval < 0:
        return -1
    return retval

cdef int H5PY_H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t 
                        file_space, hid_t xfer_plist, void* buf) nogil except -1:
    cdef herr_t retval
    retval = H5Dwrite(dset_id, mem_type, mem_space, file_space,
                        xfer_plist, buf)
    if retval < 0:
        return -1
    return retval


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

        * Hashable: Yes, unless anonymous
        * Equality: True HDF5 identity if unless anonymous
    """

    property dtype:
        """ Numpy dtype object representing the dataset type """
        def __get__(self):
            # Dataset type can't change            
            cdef TypeID tid
            if self._dtype is None:
                tid = self.get_type()
                self._dtype = tid.dtype
            return self._dtype

    property shape:
        """ Numpy-style shape tuple representing the dataspace """
        def __get__(self):
            # Shape can change (DatasetID.extend), so don't cache it
            cdef SpaceID sid
            sid = self.get_space()
            return sid.get_simple_extent_dims()

    property rank:
        """ Integer giving the dataset rank (0 = scalar) """
        def __get__(self):
            cdef SpaceID sid
            sid = self.get_space()
            return sid.get_simple_extent_ndims()

    @sync
    def _close(self):
        """ ()

            Terminate access through this identifier.  You shouldn't have to
            call this manually; Dataset objects are automatically destroyed
            when their Python wrappers are freed.
        """
        H5Dclose(self.id)

    @sync
    def read(self, SpaceID mspace not None, SpaceID fspace not None, 
                   ndarray arr_obj not None, TypeID mtype=None,
                   PropID dxpl=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj, 
             TypeID mtype=None, PropDXID dxpl=None)

            Read data from an HDF5 dataset into a Numpy array.  For maximum 
            flexibility, you can specify dataspaces for the file and the Numpy
            object. Keyword dxpl may be a dataset transfer property list.

            The provided Numpy array must be writable and C-contiguous.  If
            this is not the case, ValueError will be raised and the read will
            fail.

            It is your responsibility to ensure that the memory dataspace
            provided is compatible with the shape of the Numpy array.  Since a
            wide variety of dataspace configurations are possible, this is not
            checked.  You can easily crash Python by reading in data from too
            large a dataspace.

            The actual read is non-blocking; the array object is temporarily
            marked read-only, but attempting to mutate it in another thread
            is a bad idea.  All HDF5 API calls are locked until the read
            completes.
        """
        cdef hid_t self_id, mtype_id, mspace_id, fspace_id, plist_id
        cdef void* data
        cdef int oldflags

        if mtype is None:
            mtype = py_create(arr_obj.dtype)
        check_numpy_write(arr_obj, -1)

        self_id = self.id
        mtype_id = mtype.id
        mspace_id = mspace.id
        fspace_id = fspace.id
        plist_id = pdefault(dxpl)
        data = PyArray_DATA(arr_obj)

        arr_obj.flags &= (~NPY_WRITEABLE) # Wish-it-was-a-mutex approach
        try:
            with nogil:
                H5PY_H5Dread(self_id, mtype_id, mspace_id, fspace_id, plist_id, data)
        finally:
            arr_obj.flags |= NPY_WRITEABLE

    @sync
    def write(self, SpaceID mspace not None, SpaceID fspace not None, 
                    ndarray arr_obj not None, TypeID mtype=None,
                    PropID dxpl=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj, 
             TypeID mtype=None, PropDXID dxpl=None)

            Write data from a Numpy array to an HDF5 dataset. Keyword dxpl may 
            be a dataset transfer property list.

            The provided Numpy array must be C-contiguous.  If this is not the
            case, ValueError will be raised and the read will fail.

            The actual write is non-blocking; the array object is temporarily
            marked read-only, but attempting to mutate it in another thread
            is a bad idea.  All HDF5 API calls are locked until the write
            completes.
        """
        cdef hid_t self_id, mtype_id, mspace_id, fspace_id, plist_id
        cdef void* data
        cdef int oldflags

        if mtype is None:
            mtype = py_create(arr_obj.dtype)
        check_numpy_read(arr_obj, -1)

        self_id = self.id
        mtype_id = mtype.id
        mspace_id = mspace.id
        fspace_id = fspace.id
        plist_id = pdefault(dxpl)
        data = PyArray_DATA(arr_obj)

        arr_obj.flags &= (~NPY_WRITEABLE) # Wish-it-was-a-mutex approach
        try:
            with nogil:
                H5PY_H5Dwrite(self_id, mtype_id, mspace_id, fspace_id, plist_id, data)
        finally:
            arr_obj.flags |= NPY_WRITEABLE

    @sync
    def extend(self, tuple shape):
        """ (TUPLE shape)

            Extend the given dataset so it's at least as big as "shape".  Note 
            that a dataset may only be extended up to the maximum dimensions of 
            its dataspace, which are fixed when the dataset is created.
        """
        cdef int rank
        cdef hid_t space_id = 0
        cdef hsize_t* dims = NULL

        try:
            space_id = H5Dget_space(self.id)
            rank = H5Sget_simple_extent_ndims(space_id)

            if len(shape) != rank:
                raise TypeError("New shape length (%d) must match dataset rank (%d)" % (len(shape), rank))

            dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(shape, dims, rank)
            H5Dextend(self.id, dims)

        finally:
            efree(dims)
            if space_id:
                H5Sclose(space_id)

    @sync
    def set_extent(self, tuple shape):
        """ (TUPLE shape)

            Set the size of the dataspace to match the given shape.  If the new
            size is larger in any dimension, it must be compatible with the
            maximum dataspace size.
        """
        cdef int rank
        cdef hid_t space_id = 0
        cdef hsize_t* dims = NULL

        try:
            space_id = H5Dget_space(self.id)
            rank = H5Sget_simple_extent_ndims(space_id)

            if len(shape) != rank:
                raise TypeError("New shape length (%d) must match dataset rank (%d)" % (len(shape), rank))

            dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(shape, dims, rank)
            H5Dset_extent(self.id, dims)

        finally:
            efree(dims)
            if space_id:
                H5Sclose(space_id)


    @sync
    def get_space(self):
        """ () => SpaceID

            Create and return a new copy of the dataspace for this dataset.
        """
        return SpaceID(H5Dget_space(self.id))

    @sync
    def get_space_status(self):
        """ () => INT space_status_code

            Determine if space has been allocated for a dataset.  
            Return value is one of:

            * SPACE_STATUS_NOT_ALLOCATED
            * SPACE_STATUS_PART_ALLOCATED
            * SPACE_STATUS_ALLOCATED 
        """
        cdef H5D_space_status_t status
        H5Dget_space_status(self.id, &status)
        return <int>status

    @sync
    def get_type(self):
        """ () => TypeID

            Create and return a new copy of the datatype for this dataset.
        """
        return typewrap(H5Dget_type(self.id))

    @sync
    def get_create_plist(self):
        """ () => PropDCID

            Create an return a new copy of the dataset creation property list
            used when this dataset was created.
        """
        return propwrap(H5Dget_create_plist(self.id))

    @sync
    def get_offset(self):
        """ () => LONG offset or None

            Get the offset of this dataset in the file, in bytes, or None if
            it doesn't have one.  This is always the case for datasets which
            use chunked storage, compact datasets, and datasets for which space
            has not yet been allocated in the file.
        """
        cdef haddr_t offset
        offset = H5Dget_offset(self.id)
        if offset == HADDR_UNDEF:
            return None
        return offset

    @sync
    def get_storage_size(self):
        """ () => LONG storage_size

            Determine the amount of file space required for a dataset.  Note 
            this only counts the space which has actually been allocated; it 
            may even be zero.
        """
        return H5Dget_storage_size(self.id)



