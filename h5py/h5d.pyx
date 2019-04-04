# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
"""
    Provides access to the low-level HDF5 "H5D" dataset interface.
"""

include "config.pxi"

# Compile-time imports
from _objects cimport pdefault
from numpy cimport ndarray, import_array, PyArray_DATA, NPY_WRITEABLE
from utils cimport  check_numpy_read, check_numpy_write, \
                    convert_tuple, convert_dims, emalloc, efree
from h5t cimport TypeID, typewrap, py_create
from h5s cimport SpaceID
from h5p cimport PropID, propwrap
from _proxy cimport dset_rw

from h5py import _objects
from ._objects import phil, with_phil

# Initialization
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

IF HDF5_VERSION >= VDS_MIN_HDF5_VERSION:
    VIRTUAL = H5D_VIRTUAL
    VDS_FIRST_MISSING   = H5D_VDS_FIRST_MISSING
    VDS_LAST_AVAILABLE  = H5D_VDS_LAST_AVAILABLE

IF HDF5_VERSION >= (1, 10, 5):

    cdef class StoreInfo:
        """Represent storage information of one dataset chunk or contiguous
        dataset.

        Feature requires: 1.10.5 HDF5
        """

        cdef object index
        cdef tuple chunk_offset
        cdef unsigned filter_mask
        cdef haddr_t byte_offset
        cdef hsize_t size

        def __init__(self):
            pass

        property index:
            def __get__(self):
                """Index of written chunk.

                The None value indicates the index is not applicable/available.
                """
                return self.index

        property filter_mask:
            def __get__(self):
                """Filter mask providing a record of which filters are used.

                The default value of the mask is zero (0), indicating that
                all enabled filters are applied. A filter is skipped if the
                bit corresponding to the filter’s position in the pipeline
                (0 <= position < 32) is turned on.
                """
                return self.filter_mask

        property file_offset:
            def __get__(self):
                """File offset of the dataset chunk or contiguous dataset.

                Returns None if the chunk is not written or empty dataset.
                """
                if self.byte_offset == HADDR_UNDEF:
                    return None
                return self.byte_offset

        property size:
            def __get__(self):
                """Size of the dataset chunk or contiguous dataset in bytes."""
                return self.size

        property chunk_offset:
            def __get__(self):
                """Offset of the chunk or array index of the first
                element for contiguous datasets.

                Returns None if the chunk is not written or empty dataset.
                """
                if self.byte_offset == HADDR_UNDEF:
                    return None
                return self.chunk_offset


# === Dataset operations ======================================================

@with_phil
def create(ObjectID loc not None, object name, TypeID tid not None,
           SpaceID space not None, PropID dcpl=None, PropID lcpl=None,
           PropID dapl = None):
    """ (objectID loc, STRING name or None, TypeID tid, SpaceID space,
         PropDCID dcpl=None, PropID lcpl=None) => DatasetID

    Create a new dataset.  If "name" is None, the dataset will be
    anonymous.
    """
    cdef hid_t dsid
    cdef char* cname = NULL
    if name is not None:
        cname = name

    if cname != NULL:
        dsid = H5Dcreate2(loc.id, cname, tid.id, space.id,
                 pdefault(lcpl), pdefault(dcpl), pdefault(dapl))
    else:
        dsid = H5Dcreate_anon(loc.id, tid.id, space.id,
                 pdefault(dcpl), pdefault(dapl))
    return DatasetID(dsid)

@with_phil
def open(ObjectID loc not None, char* name, PropID dapl=None):
    """ (ObjectID loc, STRING name, PropID dapl=None) => DatasetID

    Open an existing dataset attached to a group or file object, by name.

    If specified, dapl may be a dataset access property list.
    """
    return DatasetID(H5Dopen2(loc.id, name, pdefault(dapl)))

# --- Proxy functions for safe(r) threading -----------------------------------


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
            with phil:
                if self._dtype is None:
                    tid = self.get_type()
                    self._dtype = tid.dtype
                return self._dtype

    property shape:
        """ Numpy-style shape tuple representing the dataspace """
        def __get__(self):
            # Shape can change (DatasetID.extend), so don't cache it
            cdef SpaceID sid
            with phil:
                sid = self.get_space()
                return sid.get_simple_extent_dims()

    property rank:
        """ Integer giving the dataset rank (0 = scalar) """
        def __get__(self):
            cdef SpaceID sid
            with phil:
                sid = self.get_space()
                return sid.get_simple_extent_ndims()


    @with_phil
    def read(self, SpaceID mspace not None, SpaceID fspace not None,
             ndarray arr_obj not None, TypeID mtype=None,
             PropID dxpl=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj,
             TypeID mtype=None, PropDXID dxpl=None)

            Read data from an HDF5 dataset into a Numpy array.

            It is your responsibility to ensure that the memory dataspace
            provided is compatible with the shape of the Numpy array.  Since a
            wide variety of dataspace configurations are possible, this is not
            checked.  You can easily crash Python by reading in data from too
            large a dataspace.

            If a memory datatype is not specified, one will be auto-created
            based on the array's dtype.

            The provided Numpy array must be writable and C-contiguous.  If
            this is not the case, ValueError will be raised and the read will
            fail.  Keyword dxpl may be a dataset transfer property list.
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

        dset_rw(self_id, mtype_id, mspace_id, fspace_id, plist_id, data, 1)


    @with_phil
    def write(self, SpaceID mspace not None, SpaceID fspace not None,
              ndarray arr_obj not None, TypeID mtype=None,
              PropID dxpl=None):
        """ (SpaceID mspace, SpaceID fspace, NDARRAY arr_obj,
             TypeID mtype=None, PropDXID dxpl=None)

            Write data from a Numpy array to an HDF5 dataset. Keyword dxpl may
            be a dataset transfer property list.

            It is your responsibility to ensure that the memory dataspace
            provided is compatible with the shape of the Numpy array.  Since a
            wide variety of dataspace configurations are possible, this is not
            checked.  You can easily crash Python by writing data from too
            large a dataspace.

            If a memory datatype is not specified, one will be auto-created
            based on the array's dtype.

            The provided Numpy array must be C-contiguous.  If this is not the
            case, ValueError will be raised and the read will fail.
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

        dset_rw(self_id, mtype_id, mspace_id, fspace_id, plist_id, data, 0)


    @with_phil
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


    @with_phil
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


    @with_phil
    def get_space(self):
        """ () => SpaceID

            Create and return a new copy of the dataspace for this dataset.
        """
        return SpaceID(H5Dget_space(self.id))


    @with_phil
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


    @with_phil
    def get_type(self):
        """ () => TypeID

            Create and return a new copy of the datatype for this dataset.
        """
        return typewrap(H5Dget_type(self.id))


    @with_phil
    def get_create_plist(self):
        """ () => PropDCID

            Create an return a new copy of the dataset creation property list
            used when this dataset was created.
        """
        return propwrap(H5Dget_create_plist(self.id))


    @with_phil
    def get_access_plist(self):
        """ () => PropDAID

            Create an return a new copy of the dataset access property list.
        """
        return propwrap(H5Dget_access_plist(self.id))


    @with_phil
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


    @with_phil
    def get_storage_size(self):
        """ () => LONG storage_size

            Determine the amount of file space required for a dataset.  Note
            this only counts the space which has actually been allocated; it
            may even be zero.
        """
        return H5Dget_storage_size(self.id)

    IF HDF5_VERSION >= SWMR_MIN_HDF5_VERSION:

        @with_phil
        def flush(self):
            """ no return

            Flushes all buffers associated with a dataset to disk.

            This function causes all buffers associated with a dataset to be
            immediately flushed to disk without removing the data from the cache.

            Use this in SWMR write mode to allow readers to be updated with the
            dataset changes.

            Feature requires: 1.9.178 HDF5
            """
            H5Dflush(self.id)

        @with_phil
        def refresh(self):
            """ no return

            Refreshes all buffers associated with a dataset.

            This function causes all buffers associated with a dataset to be
            cleared and immediately re-loaded with updated contents from disk.

            This function essentially closes the dataset, evicts all metadata
            associated with it from the cache, and then re-opens the dataset.
            The reopened dataset is automatically re-registered with the same ID.

            Use this in SWMR read mode to poll for dataset changes.

            Feature requires: 1.9.178 HDF5
            """
            H5Drefresh(self.id)


    IF HDF5_VERSION >= (1, 8, 11):

        def write_direct_chunk(self, offsets, bytes data, H5Z_filter_t filter_mask=H5Z_FILTER_NONE, PropID dxpl=None):
            """ (offsets, bytes data, H5Z_filter_t filter_mask=H5Z_FILTER_NONE, PropID dxpl=None)

            Writes data from a bytes array (as provided e.g. by struct.pack) directly
            to a chunk at position specified by the offsets argument.

            Feature requires: 1.8.11 HDF5
            """

            cdef hid_t dset_id
            cdef hid_t dxpl_id
            cdef hid_t space_id = 0
            cdef hsize_t *offset = NULL
            cdef size_t data_size
            cdef int rank

            dset_id = self.id
            dxpl_id = pdefault(dxpl)
            space_id = H5Dget_space(self.id)
            rank = H5Sget_simple_extent_ndims(space_id)

            if len(offsets) != rank:
                raise TypeError("offset length (%d) must match dataset rank (%d)" % (len(offsets), rank))

            try:
                offset = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
                convert_tuple(offsets, offset, rank)
                H5DOwrite_chunk(dset_id, dxpl_id, filter_mask, offset, len(data), <char *> data)
            finally:
                efree(offset)
                if space_id:
                    H5Sclose(space_id)

    IF HDF5_VERSION >= (1, 10, 5):

        @with_phil
        def get_num_chunks(self, SpaceID space=None):
            """ (SpaceID space=None) => INT num_chunks

            Retrieve the number of chunks that have nonempty intersection with a
            specified dataspace. Currently, this function only gets the number
            of all written chunks, regardless of the dataspace.

            Feature requires: 1.10.5 HDF5
            """
            cdef hsize_t num_chunks

            if space is None:
                space = self.get_space()
            H5Dget_num_chunks(self.id, space.id, &num_chunks)
            return num_chunks

        @with_phil
        def get_chunk_info(self, hsize_t index, SpaceID space=None):
            """ (hsize_t index, SpaceID space=None) => StoreInfo

            Retrieve storage information about a chunk specified by its index.

            Feature requires: 1.10.5 HDF5
            """
            cdef StoreInfo si
            cdef haddr_t byte_offset
            cdef hsize_t size
            cdef hsize_t *chunk_offset
            cdef unsigned filter_mask
            cdef hid_t space_id = 0
            cdef int rank

            if space is None:
                space_id = H5Dget_space(self.id)
            else:
                space_id = self.id

            rank = H5Sget_simple_extent_ndims(space_id)
            chunk_offset = <hsize_t*>emalloc(sizeof(hsize_t) * rank)
            H5Dget_chunk_info(self.id, space_id, index, chunk_offset,
                              &filter_mask, &byte_offset, &size)
            si = StoreInfo()
            si.index = index
            si.byte_offset = byte_offset
            si.size = size
            si.chunk_offset = convert_dims(chunk_offset, <hsize_t>rank)
            si.filter_mask = filter_mask
            efree(chunk_offset)
            if space is None:
                H5Sclose(space_id)

            return si

        @with_phil
        def get_chunk_info_by_coord(self, tuple chunk_offset not None):
            """ (TUPLE chunk_offset) => StoreInfo

            Retrieve information about a chunk specified by the array
            address of one of its dataset elements.

            Feature requires: 1.10.5 HDF5
            """
            cdef StoreInfo si
            cdef haddr_t byte_offset
            cdef hsize_t size
            cdef unsigned filter_mask
            cdef hid_t space_id = 0
            cdef int rank
            cdef hsize_t *co = NULL

            space_id = H5Dget_space(self.id)
            rank = H5Sget_simple_extent_ndims(space_id)
            H5Sclose(space_id)
            co = <hsize_t*>emalloc(sizeof(hsize_t) * rank)
            convert_tuple(chunk_offset, co, rank)
            H5Dget_chunk_info_by_coord(self.id, co, &filter_mask, &byte_offset,
                                       &size)
            efree(co)
            si = StoreInfo()
            si.index = None
            si.byte_offset = byte_offset
            si.size = size
            si.chunk_offset = chunk_offset
            si.filter_mask = filter_mask

            return si

        @with_phil
        def get_store_info(self):
            """ () => LIST[StoreInfo]

            Provide storage information for all chunks as a list of StoreInfo
            objects. This is a convenience method and not part of the H5D API.

            Feature requires: 1.10.5 HDF5
            """
            cdef list si_list = []
            cdef StoreInfo si
            cdef int rank
            cdef hid_t space_id
            cdef haddr_t byte_offset
            cdef hsize_t size
            cdef hsize_t num_chunks
            cdef hsize_t index
            cdef hsize_t *chunk_offset = NULL
            cdef unsigned filter_mask

            space_id = H5Dget_space(self.id)
            try:
                if H5Sget_simple_extent_type(space_id) == H5S_NULL:
                    return si_list

                rank = H5Sget_simple_extent_ndims(space_id)
                if H5Pget_layout(H5Dget_create_plist(self.id)) == H5D_CONTIGUOUS:
                    byte_offset = H5Dget_offset(self.id)
                    if byte_offset == HADDR_UNDEF:
                        return si_list
                    si = StoreInfo()
                    si.index = None
                    si.byte_offset = byte_offset
                    si.size = H5Dget_storage_size(self.id)
                    si.filter_mask = 0
                    si.chunk_offset = (0,) * rank
                    return [si]

                H5Dget_num_chunks(self.id, space_id, &num_chunks)
                if num_chunks == 0:
                    return si_list

                chunk_offset = <hsize_t*>emalloc(sizeof(hsize_t) * rank)
                si_list = [None] * num_chunks
                for index in range(num_chunks):
                    H5Dget_chunk_info(self.id, space_id, index, chunk_offset,
                                      &filter_mask, &byte_offset, &size)
                    si = StoreInfo()
                    si.index = index
                    si.byte_offset = byte_offset
                    si.size = size
                    si.chunk_offset = convert_dims(chunk_offset, <hsize_t>rank)
                    si.filter_mask = filter_mask
                    si_list[index] = si

                return si_list

            finally:
                H5Sclose(space_id)
                efree(chunk_offset)
