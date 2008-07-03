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
"""

# Pyrex compile-time imports
from utils cimport  require_tuple, require_list, convert_dims, convert_tuple, \
                    emalloc, efree, pybool

# Runtime imports
import h5
from h5 import DDict

cdef object lockid(hid_t id_):
    cdef SpaceID space
    space = SpaceID(id_)
    space._locked = 1
    return space

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

ALL = lockid(H5S_ALL)   # This is accepted in lieu of an actual identifier
                        # in functions like H5Dread, so wrap it.
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

def create(int class_code):
    """ (INT class_code) => SpaceID

        Create a new HDF5 dataspace object, of the given class.
        Legal values are SCALAR and SIMPLE.
    """
    return SpaceID(H5Screate(<H5S_class_t>class_code))

def create_simple(object dims_tpl, object max_dims_tpl=None):
    """ (TUPLE dims_tpl, TUPLE max_dims_tpl) => INT new_space_id

        Create a simple (slab) dataspace from a tuple of dimensions.  
        Every element of dims_tpl must be a positive integer.  

        You can optionally specify the maximum dataspace size. The 
        special value UNLIMITED, as an element of max_dims, indicates 
        an unlimited dimension.
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

        return SpaceID(H5Screate_simple(rank, dims, max_dims))

    finally:
        efree(dims)
        efree(max_dims)

# === H5S class API ===========================================================

cdef class SpaceID(ObjectID):

    """
        Represents a dataspace identifier.

        Properties:
        shape:  Numpy-style shape tuple with dimensions.
    """

    property shape:
        """ Numpy-style shape tuple representing dimensions.  () == scalar.
        """
        def __get__(self):
            return self.get_simple_extent_dims()

    def close(self):
        """ ()

            Terminate access through this identifier.  You shouldn't have to
            call this manually; dataspace objects are automatically destroyed
            when their Python wrappers are freed.
        """
        H5Sclose(self.id)

    def copy(self):
        """ () => SpaceID

            Create a new copy of this dataspace.
        """
        return SpaceID(H5Scopy(self.id))

    # === Simple dataspaces ===================================================

    def is_simple(self):
        """ () => BOOL is_simple

            Determine if an existing dataspace is "simple" (including scalar
            dataspaces). Currently all HDF5 dataspaces are simple.
        """
        return pybool(H5Sis_simple(self.id))

    def offset_simple(self, object offset=None):
        """ (TUPLE offset=None)

            Set the offset of a dataspace.  The length of the given tuple must
            match the rank of the dataspace. If None is provided (default), 
            the offsets on all axes will be set to 0.
        """
        cdef int rank
        cdef int i
        cdef hssize_t *dims
        dims = NULL

        try:
            if not H5Sis_simple(self.id):
                raise ValueError("%d is not a simple dataspace" % space_id)

            rank = H5Sget_simple_extent_ndims(self.id)
            
            require_tuple(offset, 1, rank, "offset")
            dims = <hssize_t*>emalloc(sizeof(hssize_t)*rank)
            if(offset is not None):
                convert_tuple(offset, <hsize_t*>dims, rank)
            else:
                # The HDF5 docs say passing in NULL resets the offset to 0.  
                # Instead it raises an exception.  Imagine my surprise. We'll 
                # do this manually.
                for i from 0<=i<rank:
                    dims[i] = 0

            H5Soffset_simple(self.id, dims)

        finally:
            efree(dims)

    def get_simple_extent_ndims(self):
        """ () => INT rank
            
            Determine the rank of a "simple" (slab) dataspace.
        """
        return H5Sget_simple_extent_ndims(self.id)

    def get_simple_extent_dims(self, int maxdims=0):
        """ (BOOL maxdims=False) => TUPLE shape

            Determine the shape of a "simple" (slab) dataspace.  If "maxdims" 
            is True, retrieve the maximum dataspace size instead.
        """
        cdef int rank
        cdef hsize_t* dims
        dims = NULL

        rank = H5Sget_simple_extent_dims(self.id, NULL, NULL)

        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        try:
            if maxdims:
                H5Sget_simple_extent_dims(self.id, NULL, dims)
            else:
                H5Sget_simple_extent_dims(self.id, dims, NULL)

            return convert_dims(dims, rank)

        finally:
            efree(dims)
    
    def get_simple_extent_npoints(self):
        """ () => LONG npoints

            Determine the total number of elements in a dataspace.
        """
        return H5Sget_simple_extent_npoints(self.id)

    def get_simple_extent_type(self):
        """ () => INT class_code

            Class code is either SCALAR or SIMPLE.
        """
        return <int>H5Sget_simple_extent_type(self.id)

    # === Extents =============================================================

    def extent_copy(self, SpaceID source not None):
        """ (SpaceID source)

            Replace this dataspace's extent with another's, changing its
            typecode if necessary.
        """
        H5Sextent_copy(self.id, source.id)

    def set_extent_simple(self, object dims_tpl, object max_dims_tpl=None):
        """ (TUPLE dims_tpl, TUPLE max_dims_tpl=None)

            Reset the dataspace extent via a tuple of dimensions.  
            Every element of dims_tpl must be a positive integer.  

            You can optionally specify the maximum dataspace size. The 
            special value UNLIMITED, as an element of max_dims, indicates 
            an unlimited dimension.
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

            H5Sset_extent_simple(self.id, rank, dims, max_dims)

        finally:
            efree(dims)
            efree(max_dims)

    def set_extent_none(self):
        """ ()

            Remove the dataspace extent; typecode changes to NO_CLASS.
        """
        H5Sset_extent_none(self.id)

    # === General selection operations ========================================

    def get_select_type(self):
        """ () => INT select_code

            Determine selection type.  Return values are:
            SEL_NONE:       No selection.
            SEL_ALL:        All points selected
            SEL_POINTS:     Point-by-point element selection in use
            SEL_HYPERSLABS: Hyperslab selection in use
        """
        return <int>H5Sget_select_type(self.id)

    def get_select_npoints(self):
        """ () => LONG npoints

            Determine the total number of points currently selected.  
            Works for all selection techniques.
        """
        return H5Sget_select_npoints(self.id)

    def get_select_bounds(self):
        """ () => (TUPLE start, TUPLE end)

            Determine the bounding box which exactly contains 
            the current selection.
        """
        cdef int rank
        cdef hsize_t *start
        cdef hsize_t *end
        start = NULL
        end = NULL

        rank = H5Sget_simple_extent_ndims(self.id)

        start = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        end = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

        try:
            H5Sget_select_bounds(self.id, start, end)

            start_tpl = convert_dims(start, rank)
            end_tpl = convert_dims(end, rank)
            return (start_tpl, end_tpl)

        finally:
            efree(start)
            efree(end)

    def select_all(self):
        """ ()

            Select all points in the dataspace.
        """
        H5Sselect_all(self.id)

    def select_none(self):
        """ ()

            Deselect entire dataspace.
        """
        H5Sselect_none(self.id)

    def select_valid(self):
        """ () => BOOL select_valid
            
            Determine if the current selection falls within
            the dataspace extent.
        """
        return pybool(H5Sselect_valid(self.id))

    # === Point selection functions ===========================================

    def get_select_elem_npoints(self):
        """ () => LONG npoints

            Determine the number of elements selected in point-selection mode.
        """
        return H5Sget_select_elem_npoints(self.id)

    def get_select_elem_pointlist(self):
        """ () => LIST elements_list

            Get a list of all selected elements, in point-selection mode.
            List entries are <rank>-length tuples containing point coordinates.
        """
        cdef int rank
        cdef hssize_t npoints
        cdef hsize_t *buf
        cdef int i_point
        cdef int i_entry

        npoints = H5Sget_select_elem_npoints(self.id)
        if npoints == 0:
            return []

        rank = H5Sget_simple_extent_ndims(self.id)
        
        buf = <hsize_t*>emalloc(sizeof(hsize_t)*rank*npoints)

        try:
            H5Sget_select_elem_pointlist(self.id, 0, <hsize_t>npoints, buf)

            retlist = []
            for i_point from 0<=i_point<npoints:
                tmp_tpl = []
                for i_entry from 0<=i_entry<rank:
                    tmp_tpl.append( long( buf[i_point*rank + i_entry] ) )
                retlist.append(tuple(tmp_tpl))

        finally:
            efree(buf)

        return retlist

    def select_elements(self, object coord_list, int op=H5S_SELECT_SET):
        """ (LIST coord_list, INT op=SELECT_SET)

            Select elements using a list of points.  List entries should be
            tuples containing point coordinates. A zero-length list is 
            apparently not allowed by the HDF5 library.
        """
        cdef size_t nelements   # Number of point coordinates
        cdef hsize_t *coords    # Contiguous 2D array nelements x rank x sizeof(hsize_t)

        cdef int rank
        cdef int i_point
        cdef int i_entry
        coords = NULL

        require_list(coord_list, 0, -1, "coord_list")
        nelements = len(coord_list)

        rank = H5Sget_simple_extent_ndims(self.id)

        # The docs say this should be an hsize_t**, but it seems that
        # HDF5 expects the coordinates to be a static, contiguous
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

            H5Sselect_elements(self.id, <H5S_seloper_t>op, nelements, <hsize_t**>coords)

        finally:
            efree(coords)

    # === Hyperslab selection functions =======================================

    def get_select_hyper_nblocks(self):
        """ () => LONG nblocks

            Get the number of hyperslab blocks currently selected.
        """
        return H5Sget_select_hyper_nblocks(self.id)

    def get_select_hyper_blocklist(self):
        """ () => LIST hyperslab_blocks

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

        rank = H5Sget_simple_extent_ndims(self.id)
        nblocks = H5Sget_select_hyper_nblocks(self.id)

        buf = <hsize_t*>emalloc(sizeof(hsize_t)*2*rank*nblocks)
        
        try:
            H5Sget_select_hyper_blocklist(self.id, 0, nblocks, buf)

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
    

    def select_hyperslab(self, object start, object count, object stride=None, 
                         object block=None, int op=H5S_SELECT_SET):
        """ (TUPLE start, TUPLE count, TUPLE stride=None, TUPLE block=None, 
             INT op=SELECT_SET)
         
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
        rank = H5Sget_simple_extent_ndims(self.id)

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

            H5Sselect_hyperslab(self.id, <H5S_seloper_t>op, start_array, 
                                         stride_array, count_array, block_array)

        finally:
            efree(start_array)
            efree(count_array)
            efree(stride_array)
            efree(block_array)







