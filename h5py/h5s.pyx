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
from utils cimport  require_tuple, require_list, convert_dims, convert_tuple, \
                    emalloc, efree

# Runtime imports
import h5
from h5 import DDict

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
    H5Sclose(space_id)

def create(int class_code):
    """ (INT class_code) => INT new_space_id

        Create a new HDF5 dataspace object, of the given class.  Legal values
        are CLASS_SCALAR and CLASS_SIMPLE.
    """
    return H5Screate(<H5S_class_t>class_code)

def copy(hid_t space_id):
    """ (INT space_id) => INT new_space_id

        Create a new copy of an existing dataspace.
    """
    return H5Scopy(space_id)

# === Simple dataspaces =======================================================

def create_simple(object dims_tpl, object max_dims_tpl=None):
    """ (TUPLE dims_tpl, TUPLE max_dims_tpl) => INT new_space_id

        Create a simple (slab) dataspace from a tuple of dimensions.  Every
        element of dims_tpl must be a positive integer.  You can also specify
        the maximum dataspace size, via the tuple max_dims.  The special
        integer SPACE_UNLIMITED, as an element of max_dims, indicates an
        unlimited dimension.
    """
    cdef int rank
    cdef hsize_t* dims
    cdef hsize_t* max_dims
    dims = NULL
    max_dims = NULL

    require_tuple(dims_tpl, 0, -1, "dims_tpl")
    rank = len(dims_tpl)
    require_tuple(max_dims_tpl, 1, rank, "max_dims_tpl")

    try:
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        convert_tuple(dims_tpl, dims, rank)

        if max_dims_tpl is not None:
            max_dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(max_dims_tpl, max_dims, rank)

        return H5Screate_simple(rank, dims, max_dims)

    finally:
        efree(dims)
        efree(max_dims)

def is_simple(hid_t space_id):
    """ (INT space_id) => BOOL is_simple

        Determine if an existing dataspace is "simple".  This function is
        rather silly, as all HDF5 dataspaces are (currently) simple.
    """
    return bool(H5Sis_simple(space_id))

def offset_simple(hid_t space_id, object offset=None):
    """ (INT space_id, TUPLE offset=None)

        Set the offset of a dataspace.  The length of the given tuple must
        match the rank of the dataspace. If None is provided (default), 
        the offsets on all axes will be set to 0.
    """
    cdef int rank
    cdef hssize_t *dims
    dims = NULL

    try:
        if H5Sis_simple(space_id) == 0:
            raise ValueError("%d is not a simple dataspace" % space_id)

        rank = H5Sget_simple_extent_ndims(space_id)
        
        if offset is None:
            dims = NULL
        else:
            require_tuple(offset, 0, rank, "offset")
            dims = <hssize_t*>emalloc(sizeof(hssize_t)*rank)
            convert_tuple(offset, <hsize_t*>dims, rank)

        H5Soffset_simple(space_id, dims)

    finally:
        efree(dims)

def get_simple_extent_ndims(hid_t space_id):
    """ (INT space_id) => INT rank
        
        Determine the rank of a "simple" (slab) dataspace.
    """
    return H5Sget_simple_extent_ndims(space_id)

def get_simple_extent_dims(hid_t space_id, int maxdims=0):
    """ (INT space_id, BOOL maxdims=False) => TUPLE shape

        Determine the shape of a "simple" (slab) dataspace.  If "maxdims" is
        True, retrieve the maximum dataspace size instead.
    """
    cdef int rank
    cdef hsize_t* dims
    dims = NULL

    rank = H5Sget_simple_extent_dims(space_id, NULL, NULL)

    dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
    try:
        if maxdims:
            H5Sget_simple_extent_dims(space_id, NULL, dims)
        else:
            H5Sget_simple_extent_dims(space_id, dims, NULL)

        return convert_dims(dims, rank)

    finally:
        efree(dims)
    
def get_simple_extent_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the total number of elements in a dataspace.
    """
    return H5Sget_simple_extent_npoints(space_id)

def get_simple_extent_type(hid_t space_id):
    """ (INT space_id) => INT class_code

        Class code is either CLASS_SCALAR or CLASS_SIMPLE.
    """
    return <int>H5Sget_simple_extent_type(space_id)

# === Extents =================================================================

def extent_copy(hid_t dest_id, hid_t source_id):
    """ (INT dest_id, INT source_id)

        Copy one dataspace's extent to another, changing its type if necessary.
    """
    H5Sextent_copy(dest_id, source_id)

def set_extent_simple(hid_t space_id, object dims_tpl, object max_dims_tpl=None):
    """ (INT space_id, TUPLE dims_tpl, TUPLE max_dims_tpl=None)

        Reset the dataspace extent, via a tuple of new dimensions.  Every
        element of dims_tpl must be a positive integer.  You can also specify
        the maximum dataspace size, via the tuple max_dims.  The special
        integer SPACE_UNLIMITED, as an element of max_dims, indicates an
        unlimited dimension.
    """
    cdef int rank
    cdef hsize_t* dims
    cdef hsize_t* max_dims
    dims = NULL
    max_dims = NULL

    require_tuple(dims_tpl, 0, -1, "dims_tpl")
    rank = len(dims_tpl)
    require_tuple(max_dims_tpl, 1, rank, "max_dims_tpl")

    try:
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        convert_tuple(dims_tpl, dims, rank)

        if max_dims_tpl is not None:
            max_dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(max_dims_tpl, max_dims, rank)

        H5Sset_extent_simple(space_id, rank, dims, max_dims)

    finally:
        efree(dims)
        efree(max_dims)

def set_extent_none(hid_t space_id):
    """ (INT space_id)

        Remove the dataspace extent; class changes to h5s.CLASS_NO_CLASS.
    """
    H5Sset_extent_non(space_id)

# === General selection operations ============================================

def get_select_type(hid_t space_id):
    """ (INT space_id) => INT select_code

        Determine selection type.  Return values are:
        SEL_NONE:       No selection.
        SEL_ALL:        All points selected
        SEL_POINTS:     Point-by-point element selection in use
        SEL_HYPERSLABS: Hyperslab selection in use
    """
    return <int>H5Sget_select_type(space_id)

def get_select_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the total number of points currently selected.  Works for
        all selection techniques.
    """
    return H5Sget_select_npoints(space_id)

def get_select_bounds(hid_t space_id):
    """ (INT space_id) => (TUPLE start, TUPLE end)

        Determine the bounding box which exactly contains the current
        selection.
    """
    cdef int rank
    cdef hsize_t *start
    cdef hsize_t *end
    start = NULL
    end = NULL

    rank = H5Sget_simple_extent_ndims(space_id)

    start = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
    end = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

    try:
        H5Sget_select_bounds(space_id, start, end)

        start_tpl = convert_dims(start, rank)
        end_tpl = convert_dims(end, rank)
        return (start_tpl, end_tpl)

    finally:
        efree(start)
        efree(end)

def select_all(hid_t space_id):
    """ (INT space_id)

        Select all points in the dataspace.
    """
    H5Sselect_all(space_id)

def select_none(hid_t space_id):
    """ (INT space_id)

        Deselect entire dataspace.
    """
    H5Sselect_none(space_id)

def select_valid(hid_t space_id):
    """ (INT space_id) => BOOL select_valid
        
        Determine if the current selection falls within the dataspace extent.
    """
    return bool(H5Sselect_valid(space_id))

# === Point selection functions ===============================================

def get_select_elem_npoints(hid_t space_id):
    """ (INT space_id) => LONG npoints

        Determine the number of elements selected in point-selection mode.
    """
    return H5Sget_select_elem_npoints(space_id)

def get_select_elem_pointlist(hid_t space_id):
    """ (INT space_id) => LIST elements_list

        Get a list of all selected elements, in point-selection mode.
        List entries are <rank>-length tuples containing point coordinates.
    """
    cdef int rank
    cdef hssize_t npoints
    cdef hsize_t *buf
    cdef int i_point
    cdef int i_entry

    npoints = H5Sget_select_elem_npoints(space_id)
    if npoints == 0:
        return []

    rank = H5Sget_simple_extent_ndims(space_id)
    
    buf = <hsize_t*>emalloc(sizeof(hsize_t)*rank*npoints)

    try:
        H5Sget_select_elem_pointlist(space_id, 0, <hsize_t>npoints, buf)

        retlist = []
        for i_point from 0<=i_point<npoints:
            tmp_tpl = []
            for i_entry from 0<=i_entry<rank:
                tmp_tpl.append( long( buf[i_point*rank + i_entry] ) )
            retlist.append(tuple(tmp_tpl))

    finally:
        efree(buf)

    return retlist

def select_elements(hid_t space_id, object coord_list, int op=H5S_SELECT_SET):
    """ (INT space_id, LIST coord_list, INT op=SELECT_SET)

        Select elements using a list of points.  List entries should be
        <rank>-length tuples containing point coordinates.
    """
    cdef size_t nelements       # Number of point coordinates
    cdef hsize_t *coords        # Contiguous 2D array nelements x rank x sizeof(hsize_t)

    cdef int rank
    cdef int i_point
    cdef int i_entry
    coords = NULL

    require_list(coord_list, 0, -1, "coord_list")
    nelements = len(coord_list)

    rank = H5Sget_simple_extent_ndims(space_id)

    # HDF5 expects the coordinates array to be a static, contiguous
    # array.  We'll simulate that by malloc'ing a contiguous chunk
    # and using pointer arithmetic to initialize it.
    coords = <hsize_t*>emalloc(sizeof(hsize_t)*rank*nelements)

    try:
        for i_point from 0<=i_point<nelements:

            tpl = coord_list[i_point]
            lmsg = "List element %d" % i_point
            require_tuple(tpl, 0, rank, lmsg)

            for i_entry from 0<=i_entry<rank:
                coords[(i_point*rank) + i_entry] = tpl[i_entry]

        H5Sselect_elements(space_id, <H5S_seloper_t>op, nelements, <hsize_t**>coords)

    finally:
        efree(coords)

# === Hyperslab selection functions ===========================================

def get_select_hyper_nblocks(hid_t space_id):
    """ (INT space_id) => LONG nblocks

        Get the number of hyperslab blocks currently selected.
    """
    return H5Sget_select_hyper_nblocks(space_id)

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
    nblocks = H5Sget_select_hyper_nblocks(space_id)

    buf = <hsize_t*>emalloc(sizeof(hsize_t)*2*rank*nblocks)
    
    try:
        H5Sget_select_hyper_blocklist(space_id, 0, nblocks, buf)

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
        efree(buf)

    return outlist
    

def select_hyperslab(hid_t space_id, object start, object count, 
    object stride=None, object block=None, int op=H5S_SELECT_SET):
    """ (INT space_id, TUPLE start, TUPLE count, TUPLE stride=None, 
            TUPLE block=None, INT op=SELECT_SET)
     
        Select a block region from an existing dataspace.  See the HDF5
        documentation for the meaning of the "block" and "op" keywords.
    """
    cdef int rank
    cdef hsize_t* start_array
    cdef hsize_t* count_array
    cdef hsize_t* stride_array
    cdef hsize_t* block_array

    start_array = NULL
    count_array = NULL
    stride_array = NULL
    block_array = NULL

    # Dataspace rank.  All provided tuples must match this.
    rank = H5Sget_simple_extent_ndims(space_id)

    require_tuple(start, 0, rank, "start")
    require_tuple(count, 0, rank, "count")
    require_tuple(stride, 1, rank, "stride")
    require_tuple(block, 1, rank, "block")

    try:
        start_array = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        count_array = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        convert_tuple(start, start_array, rank)
        convert_tuple(count, count_array, rank)

        if stride is not None:
            stride_array = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(stride, stride_array, rank)
        if block is not None:
            block_array = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
            convert_tuple(block, block_array, rank)

        H5Sselect_hyperslab(space_id, <H5S_seloper_t>op, start_array, 
                                     stride_array, count_array, block_array)

    finally:
        efree(start_array)
        efree(count_array)
        efree(stride_array)
        efree(block_array)

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








