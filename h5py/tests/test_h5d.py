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

import unittest
import os
import numpy

import h5py
from h5py import h5f, h5d, h5i, h5s, h5t, h5p
from h5py.h5e import DatasetError

HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/smpl_compound_chunked.hdf5')
DTYPE = numpy.dtype([('a_name','>i4'),
                     ('c_name','|S6'),
                     ('d_name', numpy.dtype( ('>i2', (5,10)) )),
                     ('e_name', '>f4'),
                     ('f_name', numpy.dtype( ('>f8', (10,)) )),
                     ('g_name', '<u1')])
SHAPE = (6,)

basearray = numpy.ndarray(SHAPE, dtype=DTYPE)
for i in range(SHAPE[0]):
    basearray[i]["a_name"] = i,
    basearray[i]["c_name"] = "Hello!"
    basearray[i]["d_name"][:] = numpy.sum(numpy.indices((5,10)),0) + i # [:] REQUIRED for some stupid reason
    basearray[i]["e_name"] = 0.96*i
    basearray[i]["f_name"][:] = numpy.array((1024.9637*i,)*10)
    basearray[i]["g_name"] = 109

class TestH5D(unittest.TestCase):

    def setUp(self):
        self.fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)
        self.did = h5d.open(self.fid, "CompoundChunked")

    def tearDown(self):
        h5d.close(self.did)
        h5f.close(self.fid)

    def test_open_close(self):
        h5d.close(self.did)
        self.assertEqual(h5i.get_type(self.did), h5i.BADID)
        self.did = h5d.open(self.fid, "CompoundChunked")
        self.assertEqual(h5i.get_type(self.did), h5i.DATASET)

        self.assertRaises(DatasetError, h5d.open, self.fid, "Something else")
        self.assertRaises(ValueError, h5d.close, -1)

    def test_read(self):
        array = numpy.ndarray(SHAPE, dtype=DTYPE)

        h5d.read(self.did, h5s.ALL, h5s.ALL, array)
        for name in DTYPE.fields:
            self.assert_(numpy.all(array[name] == basearray[name]), "%s::\n%s\n!=\n%s" % (name, array[name], basearray[name]))

        self.assertRaises(ValueError, h5d.read, -1, h5s.ALL, h5s.ALL, array)

    def test_get_space(self):
        sid = h5d.get_space(self.did)
        try:
            shape = h5s.get_simple_extent_dims(sid)
            self.assertEqual(shape, SHAPE)
        finally:
            h5s.close(sid)
        self.assertRaises(ValueError, h5d.get_space, -1)

    def test_get_space_status(self):
        status = h5d.get_space_status(self.did)
        self.assert_(status in h5d.PY_SPACE_STATUS)
        self.assertRaises(ValueError, h5d.get_space_status, -1)

    def test_get_offset(self):
        # Chunked datasets have no offset.  New test dset needed.
        self.assertRaises(ValueError, h5d.get_offset, -1)

    def test_get_storage_size(self):
        # This function can't intentionally raise an exception.
        self.assert_(h5d.get_storage_size(self.did) >= 0)

    def test_get_type(self):
        # We're not testing datatype conversion here; that's for test_h5t
        tid = h5d.get_type(self.did)
        try:
            self.assertEqual(h5i.get_type(tid), h5i.DATATYPE)
        finally:
            h5t.close(tid)
        self.assertRaises(ValueError, h5d.get_type, -1)

    def test_get_create_plist(self):
        pid = h5d.get_create_plist(self.did)
        try:
            self.assertEqual(h5i.get_type(pid), h5i.GENPROP_LST)
        finally:
            h5p.close(pid)

        self.assertRaises(ValueError, h5d.get_create_plist, -1)

    def test_py_shape(self):
        self.assertEqual(h5d.py_shape(self.did), SHAPE)
        self.assertRaises(ValueError, h5d.py_shape, -1)

    def test_py_rank(self):
        self.assertEqual(h5d.py_rank(self.did), 1)
        self.assertRaises(ValueError, h5d.py_rank, -1)

    def test_py_dtype(self):
        self.assertEqual(type(h5d.py_dtype(self.did)), numpy.dtype)
        self.assertRaises(ValueError, h5d.py_dtype, -1)
        
        












    


