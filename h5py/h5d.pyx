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
    Provides access to the low-level HDF5 "H5D" dataset interface

    Most H5D calls are unchanged.  Since dataset I/O is done with Numpy objects,
    read and write calls do not require you to explicitly define a datatype;
    the type of the given Numpy array is used instead.

    The py_* family of functions in this module provide a significantly 
    simpler interface.  They should be sufficient for nearly all dataset
    operations from Python.
"""

# Pyrex compile-time imports
from defs_c   cimport malloc, free
from h5  cimport herr_t, hid_t, size_t, hsize_t, htri_t
from h5s cimport H5Sclose, H5S_ALL, H5S_UNLIMITED
from h5t cimport H5Tclose
from h5p cimport H5P_DEFAULT
from numpy cimport ndarray, import_array

# Runtime imports
import h5
from h5 import DDict
from errors import DatasetError
import h5t
import h5s

import_array()

# === Public constants and data structures ====================================

LAYOUT_COMPACT = H5D_COMPACT
LAYOUT_CONTIGUOUS = H5D_CONTIGUOUS
LAYOUT_CHUNKED = H5D_CHUNKED
LAYOUT_MAPPER = { H5D_COMPACT: 'COMPACT', H5D_CONTIGUOUS: 'CONTIGUOUS',
                  H5D_CHUNKED: 'CHUNKED'}
LAYOUT_MAPPER = DDict(LAYOUT_MAPPER)

ALLOC_TIME_DEFAULT  = H5D_ALLOC_TIME_DEFAULT
ALLOC_TIME_LATE     = H5D_ALLOC_TIME_LATE
ALLOC_TIME_EARLY    = H5D_ALLOC_TIME_EARLY
ALLOC_TIME_INCR     = H5D_ALLOC_TIME_INCR
ALLOC_TIME_MAPPER = { H5D_ALLOC_TIME_DEFAULT: 'DEFAULT', H5D_ALLOC_TIME_LATE:'LATE',
                      H5D_ALLOC_TIME_EARLY: 'EARLY', H5D_ALLOC_TIME_INCR: 'INCR'}
ALLOC_TIME_MAPPER = DDict(ALLOC_TIME_MAPPER)

SPACE_STATUS_NOT_ALLOCATED  = H5D_SPACE_STATUS_NOT_ALLOCATED
SPACE_STATUS_PART_ALLOCATED = H5D_SPACE_STATUS_PART_ALLOCATED
SPACE_STATUS_ALLOCATED      = H5D_SPACE_STATUS_ALLOCATED
SPACE_STATUS_MAPPER = { H5D_SPACE_STATUS_NOT_ALLOCATED: 'NOT ALLOCATED', 
                        H5D_SPACE_STATUS_PART_ALLOCATED: 'PARTIALLY ALLOCATED',
                        H5D_SPACE_STATUS_ALLOCATED: 'ALLOCATED'}
SPACE_STATUS_MAPPER = DDict(SPACE_STATUS_MAPPER)

FILL_TIME_ALLOC = H5D_FILL_TIME_ALLOC
FILL_TIME_NEVER = H5D_FILL_TIME_NEVER
FILL_TIME_IFSET = H5D_FILL_TIME_IFSET
FILL_TIME_MAPPER = { H5D_FILL_TIME_ALLOC: 'ALLOCATION TIME',
                     H5D_FILL_TIME_NEVER: 'NEVER',
                     H5D_FILL_TIME_IFSET: 'IF SET' }
FILL_TIME_MAPPER = DDict(FILL_TIME_MAPPER)

FILL_VALUE_UNDEFINED    = H5D_FILL_VALUE_UNDEFINED
FILL_VALUE_DEFAULT      = H5D_FILL_VALUE_DEFAULT
FILL_VALUE_USER_DEFINED = H5D_FILL_VALUE_USER_DEFINED
FILL_VALUE_MAPPER = { H5D_FILL_VALUE_UNDEFINED: 'UNDEFINED',
                      H5D_FILL_VALUE_DEFAULT: 'DEFAULT',
                      H5D_FILL_VALUE_USER_DEFINED: 'USER-DEFINED' }
FILL_VALUE_MAPPER = DDict(FILL_VALUE_MAPPER)

# === Basic dataset operations ================================================

def create(int loc_id, char* name, hid_t type_id, hid_t space_id, hid_t plist=H5P_DEFAULT):
    """ ( INT loc_id, STRING name, INT type_id, INT space_id,
          INT plist=H5P_DEFAULT ) 
        => INT dataset_id

        Create a new dataset under an HDF5 file or group id.  Keyword plist 
        should be a dataset creation property list.

        For a friendlier version of this function, try py_create()
    """
    cdef hid_t dataset_id
    dataset_id = H5Dcreate(loc_id, name, type_id, space_id, plist)
    if dataset_id < 0:
        raise DatasetError('Failed to create dataset "%s" under %d' % (name, loc_id))
    return dataset_id

def open(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => INT dataset_id

        Open an existing dataset attached to a group or file object, by name.
    """
    cdef hid_t dset_id
    dset_id = H5Dopen(loc_id, name)
    if dset_id < 0:
        raise DatasetError('Failed to open dataset "%s" under %d' % (name, loc_id))
    return dset_id

def close(hid_t dset_id):
    """ (INT dset_id)
    """
    cdef herr_t retval
    retval = H5Dclose(dset_id)
    if retval < 0:
        raise DatasetError("Failed to close dataset %d" % dset_id)

# === Dataset I/O =============================================================

def read(hid_t dset_id, hid_t mspace_id, hid_t fspace_id, ndarray arr_obj, hid_t plist=H5P_DEFAULT):
    """ ( INT dset_id, INT mspace_id, INT fspace_id, NDARRAY arr_obj, 
          INT plist=H5P_DEFAULT )

        Read data from an HDF5 dataset into a Numpy array.  For maximum 
        flexibility, you can specify dataspaces for the file and the Numpy
        object. Keyword plist may be a dataset transfer property list.

        It is your responsibility to ensure that the memory dataspace
        provided is compatible with the shape of the Numpy array.  It is also
        up to you to ensure that the Numpy array's dtype is conversion-
        compatible with the file's datatype. 

        The given Numpy array *must* be C-contiguous, writable and aligned 
        ("NPY_BEHAVED").  This is not currently checked; anything else may
        crash Python.

        For a friendlier version of this function, try py_read_slab().
    """
    cdef hid_t mtype_id
    cdef herr_t retval
    mtype_id = 0

    try:
        mtype_id = h5t.py_dtype_to_h5t(arr_obj.dtype)
        retval = H5Dread(dset_id, mtype_id, mspace_id, fspace_id, plist, <void*>arr_obj.data)
        if retval < 0:
            raise DatasetError("Error reading from dataset %d" % dset_id)
    finally:
        if mtype_id:
            H5Tclose(mtype_id)
        
def write(hid_t dset_id, hid_t mspace_id, hid_t fspace_id, ndarray arr_obj, hid_t plist=H5P_DEFAULT):
    """ ( INT dset_id, INT mspace_id, INT fspace_id, NDARRAY arr_obj, 
          INT plist=H5P_DEFAULT )

        Write data from a Numpy array to an HDF5 dataset. Keyword plist may be 
        a dataset transfer property list.  All the caveats in h5d.read() apply 
        here as well, in particular the restrictions on the data area of the 
        Numpy array.

        For a friendlier version of this function, try py_write_slab()
    """
    cdef hid_t mtype_id
    cdef herr_t retval
    mtype_id = 0
    try:
        mtype_id = h5t.py_dtype_to_h5t(arr_obj.dtype)
        retval = H5Dwrite(dset_id, mtype_id, mspace_id, fspace_id, plist, <void*>arr_obj.data)
        if retval < 0:
            raise DatasetError("Error writing to dataset %d" % dset_id)
    finally:
        if mtype_id:
            H5Tclose(mtype_id)

# === Dataset inspection ======================================================

def get_space(hid_t dset_id):
    """ (INT dset_id) => INT space_id

        Create and return a new copy of the dataspace for this dataset.  You're
        responsible for closing it.
    """
    cdef hid_t space_id
    space_id = H5Dget_space(dset_id)
    if space_id < 0:
        raise DatasetError("Error retrieving space of dataset %d" % dset_id)
    return space_id

def get_type(hid_t dset_id):
    """ (INT dset_id) => INT type_id

        Create and return a new copy of the datatype for this dataset.You're
        responsible for closing it.
    """
    cdef hid_t type_id
    type_id = H5Dget_type(dset_id)
    if type_id < 0:
        raise DatasetError("Error retrieving type of dataset %d" % dset_id)
    return type_id

def get_create_plist(hid_t dset_id):
    """ (INT dset_id) => INT property_list_id

        Create a new copy of the dataset creation property list used when this
        dataset was created.  You're responsible for closing it.
    """
    cdef hid_t plist
    plist = H5Dget_create_plist(dset_id)
    if plist < 0:
        raise DatasetError("Error retrieving creation property list for dataset %d" % dset_id)
    return plist

# === Python extensions =======================================================

def py_create(hid_t parent_id, char* name, object data=None, object dtype=None,
              object shape=None, object chunks=None, object compression=None,
              object shuffle=False, object fletcher32=False):
    """ ( INT parent_id, STRING name, NDARRAY data=None, DTYPE dtype=None,
          TUPLE shape=None, TUPLE chunks=None, PY_INT compression=None,
          BOOL shuffle=False, BOOL fletcher32=False )
        => INT dataset_id

        Create an HDF5 dataset from Python.  You must supply *either* a Numpy
        array, in which case the dataset will be initialized to its type,
        shape, and contents, *or* both a tuple giving the dimensions and a 
        Numpy dtype object.

        This function also works for scalar arrays; providing a "shape" tuple 
        of () or a 0-dimensional array for "data" will result in a scalar 
        (h5s.CLASS_SCALAR) dataspace for the new dataset, rather than a 
        slab (h5s.CLASS_SIMPLE).

        Additional options:
        chunks          A tuple containing chunk sizes, or None
        compression     Enable DEFLATE compression at this level (0-9), or None
        shuffle         Enable/disable shuffle filter (default disabled)
        fletcher32      Enable/disable Fletcher32 error detection (default disabled)
    """
    cdef hid_t dset_id
    cdef hid_t type_id
    cdef hid_t space_id
    cdef hid_t plist
    space_id = 0
    type_id = 0
    dset_id = 0
    plist = 0

    if (data is None and not (dtype and shape)) or (data is not None and (dtype or shape)):
        raise ValueError("*Either* a Numpy array *or* both a dtype and shape must be provided.")

    if data is not None:
        shape = data.shape
        dtype = data.dtype

    try:
        if len(shape) == 0:
            space_id = h5s.create(h5s.CLASS_SCALAR)  # let's be explicit
        else:
            space_id = h5s.create_simple(shape)

        type_id = h5t.py_dtype_to_h5t(dtype)
    
        if( chunks or compression or shuffle or fletcher32):
            plist = h5p.create(H5P_DATASET_CREATE)
            if chunks:
                h5p.set_chunk(plist, chunks)    # required for compression
            if shuffle:
                h5p.set_shuffle(plist)          # must immediately precede compression
            if compression:
                h5p.set_deflate(plist, compression)
            if fletcher32:
                h5p.set_fletcher32(plist)
        else:
            plist = H5P_DEFAULT

        dset_id = create(parent_id, name, type_id, space_id, plist)

        if data is not None:
            write(dset_id, H5S_ALL, H5S_ALL, data)

    finally:
        if space_id:
            H5Sclose(space_id)
        if type_id:
            H5Tclose(type_id)
        if plist:
            H5Pclose(plist)

    return dset_id

def py_read_slab(hid_t ds_id, object start=None, object count=None, 
                 object stride=None, **kwds):
    """ (INT ds_id, TUPLE start, TUPLE count, TUPLE stride=None, **kwds)
        => NDARRAY numpy_array_out
    
        Read a hyperslab from an existing HDF5 dataset, and return it as a
        Numpy array. Dimensions are specified by:

        start:  Tuple of integers indicating the start of the selection.
                If None, the selection starts at the dataspace origin (0,0,..)
        count:  Tuple of integers indicating how many elements to read.
                If None, the selection will extend from <start> to the end of
                the dataset.  Any of the members can also be None.
        stride: Pitch of the selection.  Data points at <start> are always
                selected.  If None, 1 will be used for all axes.

        Any of the members of start/count/stride may also be None, in which
        case the origin, full extent, and a stride of 1 will be inserted
        respectively.

        Any additional keywords (**kwds) are passed to the function which maps 
        HDF5 types to Numpy dtypes; see the docstring for h5t.py_h5t_to_dtype.
        These include force_native, compound_fields, and force_string_length.

        As is customary when slicing into Numpy array objects, no dimensions 
        with length 1 are present in the returned array.  Additionally, if the
        HDF5 dataset has a scalar dataspace, then only None or empty tuples are
        allowed for start, count and stride, and the returned array will be
        0-dimensional (arr.shape == ()).
    """
    cdef hid_t mem_space
    cdef hid_t file_space
    cdef hid_t type_id
    cdef int rank
    cdef int i

    mem_space  = 0
    file_space = 0
    type_id    = 0

    try:
        # Obtain the Numpy dtype of the array
        type_id = get_type(ds_id)
        dtype = h5t.py_h5t_to_dtype(type_id, **kwds)

        # File dataspace
        file_space = get_space(ds_id)

        rank = h5s.get_simple_extent_ndims(file_space)
        file_shape = h5s.get_simple_extent_dims(file_space)

        # Validate arguments and create ones that weren't given
        if start is None:
            start = (0,)*rank
        else:
            if len(start) != rank:
                raise ValueError("Length of 'start' tuple must match dataset rank %d (got '%s')" % (rank, repr(start)))

        if count is None:
            count = []
            for i from 0<=i<rank:
                count.append(file_shape[i] - start[i])
            count = tuple(count)
        else:
            if len(count) != rank:
                raise ValueError("Length of 'count' tuple must match dataset rank %d (got '%s')" % (rank, repr(count)))

            countlist = list(count)
            for i from 0<=i<rank:
                if count[i] is None:
                    countlist[i] = file_shape[i] - start[i]
                else:
                    countlist[i] = count[i]
            count = tuple(countlist)

        if stride is not None:      # Note that h5s.select_hyperslab allows None for stride
            if len(stride) != rank:
                raise ValueError("Length of 'stride' tuple must match dataset rank %d (got '%s')" % (rank, repr(stride)))

        # Initialize Numpy array, and an HDF5 dataspace of the same size
        npy_countlist = []
        for i from 0<=i<len(count):
            if count[i] != 0 and count[i] != 1:  # No singlet dimensions
                npy_countlist.append(count[i])
        npy_count = tuple(npy_countlist)

        arr = ndarray(npy_count, dtype=dtype)
        mem_space = h5s.create_simple(npy_count)
            
        space_type = h5s.get_simple_extent_type(file_space)
        if space_type == h5s.CLASS_SIMPLE:
            h5s.select_hyperslab(file_space, start, count, stride)
            read(ds_id, mem_space, file_space, arr)
        elif space_type == h5s.CLASS_SCALAR:
            read(ds_id, H5S_ALL, H5S_ALL, arr)
        else:
            raise ValueError("Dataspace type %d is unsupported" % space_type)

    finally:
        # ignore return values on cleanup
        if mem_space:
            H5Sclose(mem_space)
        if file_space:
            H5Sclose(file_space)
        if type_id:
            H5Tclose(type_id)

    return arr

def py_write_slab(hid_t ds_id, ndarray arr_obj, object start=None, object stride=None):
    """ (INT ds_id, NDARRAY arr_obj, TUPLE start, TUPLE stride=None)

        Write the entire contents of a Numpy array into an HDF5 dataset.
        The size of the given array must fit within the dataspace of the
        HDF5 dataset.

        start:  Tuple of integers giving offset for write.  If None, the
                dataspace origin (0,0,...) will be used.
        stride: Pitch of write in dataset.  The elements of "start" are always
                selected.  If None, 1 will be used for all dimensions.

        The underlying function depends on write access to the data area of the
        Numpy array.  See the caveats in h5d.write.

        Please note that this function does absolutely no array broadcasting;
        if you want to write a (2,3) array to an (N,2,3) or (2,3,N) dataset,
        you'll have to do it yourself from Numpy.
    """
    cdef hid_t mem_space
    cdef hid_t file_space
    cdef int rank
    mem_space  = 0
    file_space = 0

    count = arr_obj.shape

    try:
        mem_space = h5s.create_simple(count)
        file_space = get_space(ds_id)

        rank = h5s.get_simple_extent_ndims(ds_id)
        file_shape = h5s.get_simple_extent_dims(ds_id)

        if len(count) != rank:
            raise ValueError("Numpy array must have same rank as the HDF5 dataset")

        if start is None:
            start = (0,)*rank
        else:
            if len(start) != rank:
                raise ValueError("Length of 'start' tuple must match dataset rank %d (got '%s')" % (rank, repr(start)))
        
        if stride is not None:
            if len(stride) != rank:
                raise ValueError("Length of 'stride' tuple must match dataset rank %d (got '%s')" % (rank, repr(stride)))

        h5s.select_hyperslab(file_space, start, count, stride)
        write(ds_id, mem_space, file_space, arr_obj)

    finally:
        # ignore return values on cleanup
        if mem_space:
            H5Sclose(mem_space)
        if file_space:
            H5Sclose(file_space)

def py_shape(hid_t dset_id):
    """ (INT dset_id) => TUPLE shape

        Obtain the dataspace of an HDF5 dataset, as a tuple.
    """
    cdef int space_id
    space_id = 0
    shape = None
    try:
        space_id = get_space(dset_id)
        shape = h5s.get_simple_extent_dims(space_id)
    finally:
        if space_id:
            H5Sclose(space_id)
    return shape

def py_rank(hid_t dset_id):
    """ (INT dset_id) => INT rank

        Obtain the rank of an HDF5 dataset.
    """
    cdef int space_id
    space_id = 0
    rank = None
    try:
        space_id = get_space(dset_id)
        rank = h5s.get_simple_extent_ndims(space_id)
    finally:
        if space_id:
            H5Sclose(space_id)
    return rank

def py_dtype(hid_t dset_id):
    """ (INT dset_id) => DTYPE numpy_dtype

        Get the datatype of an HDF5 dataset, converted to a Numpy dtype.
    """
    cdef hid_t type_id
    type_id = 0
    dtype_out = None
    try:
        type_id = get_type(dset_id)
        dtype_out = h5t.py_h5t_to_dtype(type_id)
    finally:
        if type_id:
            H5Tclose(type_id)
    return dtype_out














