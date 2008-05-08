##### Preamble block ##########################################################
# 
# This file is part of the "h5py" HDF5 Interface for Python project.
# 
# Copyright 2008 Andrew Collette
# http://software.alfven.org
# License: BSD  (See file "LICENSE" for complete license, or the URL above)
# 
##### End preamble block ######################################################

from defs_h5t cimport H5T_NATIVE_INT8
from defs_h5i cimport H5Iget_type, H5I_ATTR, H5I_BADID
from defs_h5a cimport H5Aclose, H5Acreate, H5Adelete, H5Awrite
from defs_h5p cimport H5P_DEFAULT
from defs_h5s cimport H5Screate

import os
import unittest
import tempfile

import numpy
import h5f
import h5g
import h5a
import h5s
import h5t

from errors import H5AttributeError

SCL_NAME = 'SCALAR ATTRIBUTE'
ARR_NAME = 'ARRAY ATTRIBUTE'
TEST_NAME = 'TEST ATTRIBUTE'

class TestH5A(unittest.TestCase):

    def setUp(self):
        self.fname = tempfile.mktemp(".hdf5")
        self.fid = h5f.create(self.fname, h5f.ACC_TRUNC)
        self.gid = h5g.create(self.fid, "GROUP")
        sid = h5s.create(h5s.CLASS_SCALAR)
        sid2 = h5s.create_simple((2,3))
        self.scl_attr = H5Acreate(self.gid, SCL_NAME, H5T_NATIVE_INT8, sid, H5P_DEFAULT)
        self.arr_attr = H5Acreate(self.gid, ARR_NAME, H5T_NATIVE_INT8, sid2, H5P_DEFAULT)
        h5s.close(sid2)
        h5s.close(sid)

    def tearDown(self):
        H5Aclose(self.arr_attr)
        H5Aclose(self.scl_attr)
        h5g.close(self.gid)
        h5f.close(self.fid)
        os.unlink(self.fname)
    
    def testcreate(self):
        sid = h5s.create(h5s.CLASS_SCALAR)
        aid = h5a.create(self.gid, TEST_NAME, H5T_NATIVE_INT8, sid)
        self.assert_(H5Iget_type(aid) == H5I_ATTR)
        H5Aclose(aid)
        H5Adelete(self.gid, TEST_NAME)
        h5s.close(sid)

    def test_open_idx(self):
        aid = h5a.open_idx(self.gid, 0)
        self.assert_(h5a.get_name(aid) == SCL_NAME)
        H5Aclose(aid)
        aid = h5a.open_idx(self.gid, 1)
        self.assert_(h5a.get_name(aid) == ARR_NAME)
        H5Aclose(aid)

        self.assertRaises(H5AttributeError, h5a.open_idx, self.gid, 2)

    def test_open_name(self):
        aid = h5a.open_name(self.gid, SCL_NAME)
        self.assert_(H5Iget_type(aid) == H5I_ATTR)
        H5Aclose(aid)

    def test_close(self):
        sid = H5Screate(h5s.CLASS_SCALAR)
        aid = H5Acreate(self.gid, TEST_NAME, H5T_NATIVE_INT8, sid, H5P_DEFAULT)
        h5s.close(sid)
        self.assert_(H5Iget_type(aid) == H5I_ATTR)
        h5a.close(aid)
        self.assert_(H5Iget_type(aid) == H5I_BADID)

    def test_delete(self):
        cdef char foo
        foo = 1

        sid = H5Screate(h5s.CLASS_SCALAR)
        aid = H5Acreate(self.gid, TEST_NAME, H5T_NATIVE_INT8, sid, H5P_DEFAULT)
        h5s.close(sid)
        self.assert_(H5Iget_type(aid) == H5I_ATTR)

        retval = H5Awrite(aid, H5T_NATIVE_INT8, &foo)
        assert retval >= 0
        
        H5Aclose(aid)

        aid = h5a.open_name(self.gid, TEST_NAME)
        h5a.close(aid)

        h5a.delete(self.gid, TEST_NAME)
        self.assertRaises(H5AttributeError, h5a.open_name, self.gid, TEST_NAME)

    def test_read(self):

        cdef char foo
        foo = 42
        sid = H5Screate(h5s.CLASS_SCALAR)
        aid = H5Acreate(self.gid, TEST_NAME, H5T_NATIVE_INT8, sid, H5P_DEFAULT)
        h5s.close(sid)
        self.assert_(H5Iget_type(aid) == H5I_ATTR)

        retval = H5Awrite(aid, H5T_NATIVE_INT8, &foo)
        assert retval >= 0

        a = numpy.ndarray((1,),dtype=h5t.py_h5t_to_dtype(H5T_NATIVE_INT8))
        h5a.read(aid, a)

        self.assert_(a[0] == 42)
        H5Aclose(aid)
        H5Adelete(self.gid, TEST_NAME)

        

        

        










