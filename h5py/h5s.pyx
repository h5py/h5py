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
from defs_c   cimport malloc, free, memcpy
from h5  cimport herr_t, htri_t, hid_t, size_t, hsize_t, hssize_t
from utils cimport tuple_to_dims, dims_to_tuple, emalloc

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

ALL       = H5S_ALL
UNLIMITED = H5S_UNLIMITED

#enum H5S_class_t
NO_CLASS = H5S_NO_CLASS
SCALAR   = H5S_SCALAR
SIMPLE   = H5S_SIMPLE

#enum H5S_sel_type
SEL_ERROR       = H5S_SEL_ERROR
SEL_NONE        = H5S_SEL_NONE
SEL_POINTS      = H5S_SEL_POINTS
SEL_HYPERSLABS  = H5S_SEL_HYPERSLABS
SEL_ALL         = H5S_SEL_ALL

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

def copy(hid_t space_id):
    """ (INT space_id) => INT new_space_id

        Create a new copy of an existing dataspace.
    """
    cdef hid_t retval
    retval = H5Scopy(space_id)
    if retval < 0:
        raise DataspaceError("Failed to copy dataspace %d" % space_id)
    return retval

# === Simple dataspaces =======================================================

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

def is_simple(hid_t space_id):
    """ (INT space_id) => BOOL is_simple

        Determine if an existing dataspace is "simple".  This function is
        rather silly, as all HDF5 dataspaces are (currently) simple.
    """
    cdef htri_t retval
    retval = H5Sis_simple(space_id)
    if retval < 0:
        raise DataspaceError("Failed to determine simplicity of dataspace %d" % space_id)
    return bool(retval)

def offset_simple(hid_t space_id, object offset=None):
    """ (INT space_id, TUPLE offset=None)

        Set the offset of a dataspace.  The length of the given tuple must
        match the rank of the dataspace; ValueError will be raised otherwise.
        If None is provided (default), the offsets on all axes will be set to 0.
    """
    cdef htri_t simple
    cdef int rank
    cdef hssize_t *dims
    cdef herr_t retval
    dims = NULL

    try:
        simple = H5Sis_simple(space_id)
        if simple < 0:
            raise DataspaceError("%d is not a simple dataspace" % space_id)

        if offset is None:
            dims = NULL
        else:
            rank = H5Sget_simple_extent_ndims(space_id)
            if rank < 0:
                raise DataspaceError("Can't determine rank of dataspace %d" % space_id)

            if len(offset) != rank:
                raise ValueError("Length of offset tuple must match dataspace rank %d (got %s)" % (rank, repr(offset)))

            # why the hell are they using hssize_t?
            dims = <hssize_t*>malloc(sizeof(hssize_t)*rank)
            for i from 0<=i<rank:
                dims[i] = offset[i]

        retval = H5Soffset_simple(space_id, dims)
        if retval < 0:
            raise DataspaceError("Failed to set offset on dataspace %d" % space_id)
    finally:
        if dims != NULL:
            free(dims)

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
    
def get_simple_extent_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the total number of elements in a dataspace.
    """
    cdef hssize_t retval
    retval = H5Sget_simple_extent_npoints(space_id)
    if retval < 0:
        raise DataspaceError("Failed to determine number of points in dataspace %d" % space_id)
    return retval

def get_simple_extent_type(hid_t space_id):
    """ (INT space_id) => INT class_code

        Class code is either CLASS_SCALAR or CLASS_SIMPLE.
    """
    cdef int retval
    retval = <int>H5Sget_simple_extent_type(space_id)
    if retval < 0:
        raise DataspaceError("Can't determine type of dataspace %d" % space_id)
    return retval

# === Extents =================================================================

def extent_copy(hid_t dest_id, hid_t source_id):
    """ (INT dest_id, INT source_id)

        Copy one dataspace's extent to another, changing its type if necessary.
    """
    cdef herr_t retval
    retval = H5Sextent_copy(dest_id, source_id)
    if retval < 0:
        raise DataspaceError("Can't copy extent (%d to %d)" % (source_id, dest_id))

def set_extent_simple(hid_t space_id, object dims_tpl, object max_dims_tpl=None):
    """ (INT space_id, TUPLE dims_tpl, TUPLE max_dims_tpl=None)

        Reset the dataspace extent, via a tuple of new dimensions.  Every
        element of dims_tpl must be a positive integer.  You can also specify
        the maximum dataspace size, via the tuple max_dims.  The special
        integer h5s.SPACE_UNLIMITED, as an element of max_dims, indicates an
        unlimited dimension.
    """
    cdef int rank
    cdef hsize_t* dims
    cdef hsize_t* max_dims
    cdef herr_t retval
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

        retval = H5Sset_extent_simple(space_id, rank, dims, max_dims)

        if retval < 0:
            raise DataspaceError("Failed to reset extent to %s on space %d" % (str(dims_tpl), space_id))
    finally:
        if dims != NULL:
            free(dims)
        if max_dims != NULL:
            free(max_dims)

def set_extent_none(hid_t space_id):
    """ (INT space_id)

        Remove the dataspace extent; class changes to h5s.CLASS_NO_CLASS.
    """
    cdef herr_t retval
    retval = H5Sset_extent_non(space_id)
    if retval < 0:
        raise DataspaceError("Failed to remove extent from dataspace %d" % space_id)

# === General selection operations ============================================

def get_select_type(hid_t space_id):
    """ (INT space_id) => INT select_code

        Determine selection type.  Return values are:
        SEL_NONE:       No selection.
        SEL_ALL:        All points selected
        SEL_POINTS:     Point-by-point element selection in use
        SEL_HYPERSLABS: Hyperslab selection in use
    """
    cdef int sel_code
    sel_code = <int>H5Sget_select_type(space_id)
    if sel_code < 0:
        raise DataspaceError("Failed to determine selection type of dataspace %d" % space_id)
    return sel_code

def get_select_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the total number of points currently selected.  Works for
        all selection techniques.
    """
    cdef hssize_t retval
    retval = H5Sget_select_npoints(space_id)
    if retval < 0:
        raise DataspaceError("Failed to determine number of selected points in dataspace %d" % space_id)
    return retval

def get_select_bounds(hid_t space_id):
    """ (INT space_id) => (TUPLE start, TUPLE end)

        Determine the bounding box which exactly contains the current
        selection.
    """
    cdef int rank
    cdef herr_t retval
    cdef hsize_t *start
    cdef hsize_t *end
    start = NULL
    end = NULL

    rank = H5Sget_simple_extent_ndims(space_id)
    if rank < 0:
        raise DataspaceError("Failed to enumerate dimensions of %d for bounding box." % space_id)

    start = <hsize_t*>malloc(sizeof(hsize_t)*rank)
    end = <hsize_t*>malloc(sizeof(hsize_t)*rank)

    try:
        retval = H5Sget_select_bounds(space_id, start, end)
        if retval < 0:
            raise DataspaceError("Failed to determine bounding box for space %d" % space_id)

        start_tpl = dims_to_tuple(start, rank)
        end_tpl = dims_to_tuple(end, rank)
        if start_tpl == None or end_tpl == None:
            raise RuntimeError("Failed to construct return tuples.")

    finally:
        free(start)
        free(end)

    return (start_tpl, end_tpl)

def select_all(hid_t space_id):
    """ (INT space_id)

        Select all points in the dataspace.
    """
    cdef herr_t retval
    retval = H5Sselect_all(space_id)
    if retval < 0:
        raise DataspaceError("select_all failed on dataspace %d" % space_id)

def select_none(hid_t space_id):
    """ (INT space_id)

        Deselect entire dataspace.
    """
    cdef herr_t retval
    retval = H5Sselect_none(space_id)
    if retval < 0:
        raise DataspaceError("select_none failed on dataspace %d" % space_id)

def select_valid(hid_t space_id):
    """ (INT space_id) => BOOL select_valid
        
        Determine if the current selection falls within the dataspace extent.
    """
    cdef htri_t retval
    retval = H5Sselect_valid(space_id)
    if retval < 0:
        raise DataspaceError("Failed to determine selection status on dataspace %d" % space_id)
    return bool(retval)

# === Point selection functions ===============================================

def get_select_elem_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the number of elements selected in point-selection mode.
    """
    cdef hssize_t retval
    retval = H5Sget_select_elem_npoints(space_id)
    if retval < 0:
        raise DataspaceError("Failed to count element-selection npoints in space %d" % space_id)
    return retval

def get_select_elem_pointlist(hid_t space_id):
    """ (INT space_id) => LIST elements_list

        Get a list of all selected elements, in point-selection mode.
        List entries <rank>-length tuples containing point coordinates.
    """
    cdef herr_t retval
    cdef int rank
    cdef hssize_t npoints
    cdef hsize_t *buf
    cdef int i_point
    cdef int i_entry

    npoints = H5Sget_select_elem_npoints(space_id)
    if npoints < 0:
        raise DataspaceError("Failed to enumerate points for pointlist, space %d" % space_id)
    elif npoints == 0:
        return []

    rank = H5Sget_simple_extent_ndims(space_id)
    if rank < 0:
        raise DataspaceError("Failed to determine rank of space %d" % space_id)
    
    buf = <hsize_t*>malloc(sizeof(hsize_t)*rank*npoints)

    try:
        retval = H5Sget_select_elem_pointlist(space_id, 0, <hsize_t>npoints, buf)
        if retval < 0:
            raise DataspaceError("Failed to retrieve pointlist for dataspace %d" % space_id)

        retlist = []
        for i_point from 0<=i_point<npoints:
            tmp_tpl = []
            for i_entry from 0<=i_entry<rank:
                tmp_tpl.append( long( buf[i_point*rank + i_entry] ) )
            retlist.append(tuple(tmp_tpl))

    finally:
        free(buf)

    return retlist

def select_elements(hid_t space_id, object coord_list, int op=H5S_SELECT_SET):
    """ (INT space_id, LIST coord_list, INT op=SELECT_SET)

        Select elements using a list of points.  List entries should be
        <rank>-length tuples containing point coordinates.
    """
    cdef herr_t retval          # Result of API call
    cdef size_t nelements       # Number of point coordinates
    cdef hsize_t *coords        # Contiguous 2D array nelements x rank x sizeof(hsize_t)
    cdef size_t element_size    # Size of a point record: sizeof(hsize_t)*rank

    cdef int rank
    cdef int i_point
    cdef int i_entry
    coords = NULL

    rank = H5Sget_simple_extent_ndims(space_id)
    if rank < 0:
        raise DataspaceError("Failed to determine rank of space %d" % space_id)

    nelements = len(coord_list)
    element_size = sizeof(hsize_t)*rank

    # HDF5 docs say this has to be a contiguous 2D array
    coords = <hsize_t*>malloc(element_size*nelements)

    try:
        for i_point from 0<=i_point<nelements:

            tpl = coord_list[i_point]
            if len(tpl) != rank:
                raise ValueError("All coordinate entries must be length-%d" % rank)

            for i_entry from 0<=i_entry<rank:
                coords[(i_point*rank) + i_entry] = tpl[i_entry]

        retval = H5Sselect_elements(space_id, <H5S_seloper_t>op, nelements, <hsize_t**>coords)
        if retval < 0:
            raise DataspaceError("Failed to select point list on dataspace %d" % space_id)
    finally:
        if coords != NULL:
            free(coords)

# === Hyperslab selection functions ===========================================

def get_select_hyper_nblocks(hid_t space_id):
    """ (INT space_id) => LONG nblocks

        Get the number of hyperslab blocks currently selected.
    """
    cdef hssize_t nblocks
    nblocks = H5Sget_select_hyper_nblocks(space_id)
    if nblocks < 0:
        raise DataspaceError("Failed to enumerate selected hyperslab blocks in space %d" % space_id)
    return nblocks

def get_select_hyper_blocklist(hid_t space_id):
    """ (INT space_id) => LIST hyperslab_blocks

        Get a Python list containing selected hyperslab blocks.
        List entries are 2-tuples in the form:
            ( corner_coordinate, opposite_coordinate )
        where corner_coordinate and opposite_coordinate are <rank>-length
        tuples.
    """
    cdef hssize_t nblocks
    cdef herr_t retval
    cdef hsize_t *buf

    cdef int rank
    cdef int i_block
    cdef int i_entry

    rank = H5Sget_simple_extent_ndims(space_id)
    if rank < 0:
        raise DataspaceError("Failed to determine rank of space %d" % space_id)

    nblocks = H5Sget_select_hyper_nblocks(space_id)
    if nblocks < 0:
        raise DataspaceError("Failed to enumerate block selection on space %d" % space_id)

    buf = <hsize_t*>malloc(sizeof(hsize_t)*2*rank*nblocks)
    
    try:
        retval = H5Sget_select_hyper_blocklist(space_id, 0, nblocks, buf)
        if retval < 0:
            raise DataspaceError("Failed to retrieve list of hyperslab blocks from space %d" % space_id)

        outlist = []
        for i_block from 0<=i_block<nblocks:
            corner_list = []
            opposite_list = []
            for i_entry from 0<=i_entry<(2*rank):
                entry = long(buf[ i_block*(2*rank) + i_entry])
                if i_entry < rank:
                    corner_list.append(entry)
                else:
                    opposite_list.append(entry)
            outlist.append( (tuple(corner_list), tuple(opposite_list)) )
    finally:
        free(buf)

    return outlist
    

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

# === Python extensions =======================================================

PY_CLASS = DDict({H5S_ALL: 'ALL', H5S_UNLIMITED: 'UNLIMITED',
            H5S_NO_CLASS: 'NO CLASS', H5S_SCALAR: 'CLASS SCALAR',
            H5S_SIMPLE: 'CLASS SIMPLE'})
PY_SEL = DDict({ H5S_SEL_ERROR: 'SELECTION ERROR', H5S_SEL_NONE: 'SELECT NONE', 
            H5S_SEL_POINTS: 'POINT SELECTION', 
            H5S_SEL_HYPERSLABS: 'HYPERSLAB SELECTION',
            H5S_SEL_ALL: 'SELECT ALL' })

PY_SELECT = DDict({ H5S_SELECT_NOOP: 'NO-OP SELECT', 
                    H5S_SELECT_SET: 'SET SELECT', 
                    H5S_SELECT_OR: 'OR SELECT',
                    H5S_SELECT_AND: 'AND SELECT', H5S_SELECT_XOR: 'XOR SELECT', 
                    H5S_SELECT_NOTB: 'NOTB SELECT', H5S_SELECT_NOTA: 'NOTA SELECT', 
                    H5S_SELECT_APPEND: 'APPEND SELECTION',
                    H5S_SELECT_PREPEND: 'PREPEND SELECTION', 
                    H5S_SELECT_INVALID: 'INVALID SELECTION' })








