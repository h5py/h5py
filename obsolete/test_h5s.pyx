##### Preamble block ##########################################################
# 
# This file is part of the "h5py" HDF5 Interface for Python project.
# 
# Copyright 2008 Andrew Collette
# http://software.alfven.org
# License: BSD  (See file "LICENSE" for complete license, or the URL above)
# 
##### End preamble block ######################################################

from defs_c   cimport malloc, free
from defs_h5  cimport hsize_t
from defs_h5i cimport H5Iget_type, H5I_BADID, H5I_DATASPACE
from defs_h5s cimport H5Sget_simple_extent_ndims, H5Sget_simple_extent_dims, \
                      H5Sclose, H5Screate_simple, H5Sget_select_bounds, \
                      H5Sselect_none

import unittest
import h5s
from errors import DataspaceError

cdef int NDIMS
NDIMS = 3
DIMS = (10,13,24)
SELECT_START        = (1, 1, 5)
SELECT_LEN          = (2, 3, 4)
SELECT_STRIDE       = (1, 1, 2)
SELECT_BBOX_START   = (1, 1, 5)
SELECT_BBOX_END     = (2, 3, 11)

class TestH5S(unittest.TestCase):

    def setUp(self):

        cdef hsize_t *dims
        dims = <hsize_t*>malloc(sizeof(hsize_t)*NDIMS)
        for idx, val in enumerate(DIMS):
            dims[idx] = val
        self.sid = H5Screate_simple(NDIMS, dims, NULL)
        free(dims)

    def tearDown(self):
        H5Sclose(self.sid)

    def test_close(self):

        self.assert_(H5Iget_type(self.sid) == H5I_DATASPACE)
        h5s.close(self.sid)
        self.assert_(H5Iget_type(self.sid) == H5I_BADID)
        self.setUp()

    def test_create(self):

        cdef hsize_t *dims
        sid = h5s.create_simple(DIMS)

        self.assert_(H5Sget_simple_extent_ndims(sid) == NDIMS)
        dims = <hsize_t*>malloc(sizeof(hsize_t)*NDIMS)
        H5Sget_simple_extent_dims(sid, dims, NULL)
        for idx, val in enumerate(DIMS):
            self.assert_( dims[idx] == val )
        free(dims)
        H5Sclose(sid)

    def test_ndims(self):
        self.assert_(h5s.get_simple_extent_ndims(self.sid) == NDIMS)
        self.assertRaises(DataspaceError, h5s.get_simple_extent_ndims, -1)

    def test_dims(self):
        self.assert_(h5s.get_simple_extent_dims(self.sid) == DIMS)
        self.assertRaises(DataspaceError, h5s.get_simple_extent_dims, -1)

    def test_hyperslab(self):

        cdef hsize_t *start
        cdef hsize_t *end

        self.assertRaises(DataspaceError, h5s.select_hyperslab, self.sid, (1,), (1,) )
        self.assertRaises(DataspaceError, h5s.select_hyperslab, self.sid, SELECT_START, SELECT_LEN, SELECT_STRIDE[0:2] )
        self.assertRaises(DataspaceError, h5s.select_hyperslab, self.sid, SELECT_START, SELECT_LEN[0:2], SELECT_STRIDE )
        self.assertRaises(DataspaceError, h5s.select_hyperslab, self.sid, SELECT_START[0:2], SELECT_LEN, SELECT_STRIDE )

        H5Sselect_none(self.sid)

        start = <hsize_t*>malloc(sizeof(hsize_t)*NDIMS)
        end = <hsize_t*>malloc(sizeof(hsize_t)*NDIMS)

        h5s.select_hyperslab(self.sid, SELECT_START, SELECT_LEN, SELECT_STRIDE)
        H5Sget_select_bounds(self.sid, start, end)

        for idx in range(NDIMS):
            self.assert_( start[idx] == SELECT_BBOX_START[idx] )
            self.assert_( end[idx] == SELECT_BBOX_END[idx] )
        free(start)
        free(end)

        H5Sselect_none(self.sid)

        
        





























        
