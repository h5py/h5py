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
    Low-level interface to the "H5S" family of data-space functions.

    This module is incomplete; it currently only implements hyperslab and 
    scalar operations.
"""

# Pyrex compile-time imports
from defs_c   cimport malloc, free
from h5  cimport herr_t, hid_t, size_t, hsize_t
from utils cimport tuple_to_dims, dims_to_tuple

# Runtime imports
import h5
from h5 import DDict
from errors import DataspaceError


# === Public constants and data structures ====================================

#enum H5S_seloper_t:
SELECT_NOOP     = H5S_SELECT_NOOP
SELECT_SET      = H5S_SELECT_SET      
SELECT_OR       = H5S_SELECT_OR
SELECT_AND      = H5S_SELECT_AND
SELECT_XOR      = H5S_SELECT_XOR
SELECT_NOTB     = H5S_SELECT_NOTB
SELECT_NOTA     = H5S_SELECT_NOTA
SELECT_APPEND   = H5S_SELECT_APPEND
SELECT_PREPEND  = H5S_SELECT_PREPEND
SELECT_INVALID  = H5S_SELECT_INVALID 
SELECT_MAPPER = {H5S_SELECT_NOOP: 'NO-OP', H5S_SELECT_SET: 'SET', H5S_SELECT_OR: 'OR',
                 H5S_SELECT_AND: 'AND', H5S_SELECT_XOR: 'XOR', H5S_SELECT_NOTB: 'NOTB',
                 H5S_SELECT_NOTA: 'NOTA', H5S_SELECT_APPEND: 'APPEND',
                 H5S_SELECT_PREPEND: 'PREPEND', H5S_SELECT_INVALID: 'INVALID' }
SELECT_MAPPER = DDict(SELECT_MAPPER)

SPACE_ALL       = H5S_ALL
SPACE_UNLIMITED = H5S_UNLIMITED
SPACE_MAPPER = DDict({H5S_ALL: 'ALL', H5S_UNLIMITED: 'UNLIMITED'})

#enum H5S_class_t
CLASS_NO_CLASS = H5S_NO_CLASS
CLASS_SCALAR   = H5S_SCALAR
CLASS_SIMPLE   = H5S_SIMPLE
CLASS_COMPLEX  = H5S_COMPLEX
CLASS_MAPPER = {H5S_NO_CLASS: 'NO CLASS', H5S_SCALAR: 'SCALAR',
                H5S_SIMPLE: 'SIMPLE', H5S_COMPLEX: 'COMPLEX' }
CLASS_MAPPER = DDict(CLASS_MAPPER)

# === Basic dataspace operations ==============================================

def close(hid_t space_id):
    """ (INT space_id)
    """
    cdef herr_t retval

    retval = H5Sclose(space_id)
    if retval < 0:
        raise DataspaceError("Failed to close dataspace %d" % space_id)

def create(int class_code):
    """ (INT class_code) => INT new_space_id

        Create a new HDF5 dataspace object, of the given class.  Legal values
        are CLASS_SCALAR and CLASS_SIMPLE.
    """
    cdef hid_t retval
    retval = H5Screate(<H5S_class_t>class_code)
    if retval < 0:
        raise DataspaceError("Failed to create dataspace of class %d" %d)
    return retval

def create_simple(object dims_tpl, object max_dims_tpl=None):
    """ (TUPLE dims_tpl, TUPLE max_dims_tpl) => INT new_space_id

        Create a simple (slab) dataspace from a tuple of dimensions.  Every
        element of dims_tpl must be a positive integer.  You can also specify
        the maximum dataspace size, via the tuple max_dims.  The special
        integer h5s.SPACE_UNLIMITED, as an element of max_dims, indicates an
        unlimited dimension.
    """
    cdef hid_t space_id
    cdef int rank
    cdef hsize_t* dims
    cdef hsize_t* max_dims
    dims = NULL
    max_dims = NULL

    rank = len(dims_tpl)
    if max_dims_tpl is not None and len(max_dims_tpl) != rank:
        raise ValueError("Dims/max dims tuples must be the same rank: %s vs %s" % (repr(dims_tpl),repr(max_dims_tpl)))

    try:
        dims = tuple_to_dims(dims_tpl)
        if dims == NULL:
            raise ValueError("Bad dimensions tuple: %s" % repr(dims_tpl))

        if max_dims_tpl is not None:
            max_dims = tuple_to_dims(max_dims_tpl)
            if max_dims == NULL:
                raise ValueError("Bad max dimensions tuple: %s" % repr(max_dims_tpl))

        space_id = H5Screate_simple(rank, dims, max_dims)

        if space_id < 0:
            raise DataspaceError("Failed to create dataspace with dimensions %s" % str(dims_tpl))
    finally:
        if dims != NULL:
            free(dims)
        if max_dims != NULL:
            free(max_dims)

    return space_id

def get_simple_extent_ndims(hid_t space_id):
    """ (INT space_id) => INT rank
        
        Determine the rank of a "simple" (slab) dataspace.
    """
    cdef int ndims
    ndims = H5Sget_simple_extent_ndims(space_id)
    if ndims < 0:
        raise DataspaceError("Failed to retrieve dimension info for dataspace %d" % space_id)

    return ndims


def get_simple_extent_dims(hid_t space_id, int maxdims=0):
    """ (INT space_id, BOOL maxdims=False) => TUPLE shape

        Determine the shape of a "simple" (slab) dataspace.  If "maxdims" is
        True, retrieve the maximum dataspace size instead.
    """
    cdef int rank
    cdef hsize_t* dims
    dims = NULL
    dims_tpl = None

    rank = H5Sget_simple_extent_dims(space_id, NULL, NULL)
    if rank < 0:
        raise DataspaceError("Failed to retrieve dimension info for dataspace %d" % space_id)

    dims = <hsize_t*>malloc(sizeof(hsize_t)*rank)
    try:
        if maxdims:
            rank = H5Sget_simple_extent_dims(space_id, NULL, dims)
        else:
            rank = H5Sget_simple_extent_dims(space_id, dims, NULL)
        if rank < 0:
            raise DataspaceError("Failed to retrieve dimension info for dataspace %d" % space_id)

        dims_tpl = dims_to_tuple(dims, rank)
        if dims_tpl is None:
            raise DataspaceError("Can't unwrap dimensions on dataspace %d rank %d" % (space_id, rank))
    finally:
        if dims != NULL:
            free(dims)

    return dims_tpl
    
def get_simple_extent_type(hid_t space_id):
    """ (INT space_id) => INT class_code

        Class code is either CLASS_SCALAR or CLASS_SIMPLE.
    """
    cdef int retval
    retval = <int>H5Sget_simple_extent_type(space_id)
    if retval < 0:
        raise DataspaceError("Can't determine type of dataspace %d" % space_id)
    return retval

# === Dataspace manipulation ==================================================

def select_hyperslab(hid_t space_id, object start, object count, 
    object stride=None, object block=None, int op=H5S_SELECT_SET):
    """ (INT space_id, TUPLE start, TUPLE count, TUPLE stride=None, 
            TUPLE block=None, INT op=SELECT_SET)
     
        Select a block region from an existing dataspace.  See the HDF5
        documentation for the meaning of the "block" and "op" keywords.
    """
    cdef herr_t retval
    cdef int rank
    cdef hsize_t* start_array
    cdef hsize_t* count_array
    cdef hsize_t* stride_array
    cdef hsize_t* block_array

    start_array = NULL
    count_array = NULL
    stride_array = NULL
    block_array = NULL

    rank = get_simple_extent_ndims(space_id)
    if len(start) != rank:
        raise DataspaceError('Dimensions of input "%s" must match rank of dataspace (%d)' % (repr(start), rank))

    if len(count) != rank:
        raise DataspaceError("Dimensions of all arguments must be the same and of rank %d" % rank)

    if stride is not None:
        if len(stride) != rank:
            raise DataspaceError("Dimensions of all arguments must be the same and of rank %d" % rank)
    
    if block is not None:
        if len(block) != rank:
            raise DataspaceError("Dimensions of all arguments must be the same and of rank %d" % rank)

    try:
        start_array = tuple_to_dims(start)
        if start_array == NULL:
            raise ValueError("Invalid start tuple: %s" % repr(start))

        count_array = tuple_to_dims(count)
        if count_array == NULL:
            raise ValueError("Invalid count tuple: %s" % repr(count))

        if stride is not None:
            stride_array = tuple_to_dims(stride)
            if stride_array == NULL:
                raise ValueError("Invalid stride tuple: %s" % repr(stride))

        if block is not None:
            block_array = tuple_to_dims(block)
            if block_array == NULL:
                raise ValueError("Invalid block tuple: %s" % repr(block))

        retval = H5Sselect_hyperslab(space_id, <H5S_seloper_t>op, start_array, 
                                     stride_array, count_array, block_array)
        if retval < 0:
            raise DataspaceError("Failed to select hyperslab on dataspace %d" % space_id)
    finally:
        if start_array != NULL:
            free(start_array)
        if count_array != NULL:
            free(count_array)
        if stride_array != NULL:
            free(stride_array)
        if block_array != NULL:
            free(block_array)








