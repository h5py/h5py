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
from __future__ import with_statement

import unittest
import os
import numpy
from common import HCopy

import h5py
from h5py import h5f, h5d, h5i, h5s, h5t, h5p
from h5py.h5 import H5Error

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
        self.dset = h5d.open(self.fid, "CompoundChunked")

    def tearDown(self):
        self.dset.close()
        self.fid.close()

    def test_open_close(self):
        with HCopy(HDFNAME) as fid:
            dset = h5d.open(fid, "CompoundChunked")
            self.assertEqual(h5i.get_type(dset), h5i.DATASET)
            dset.close()
            self.assertEqual(h5i.get_type(dset), h5i.BADID)

    def test_read(self):
        array = numpy.ndarray(SHAPE, dtype=DTYPE)

        self.dset.read(h5s.ALL, h5s.ALL, array)
        for name in DTYPE.fields:
            self.assert_(numpy.all(array[name] == basearray[name]), "%s::\n%s\n!=\n%s" % (name, array[name], basearray[name]))

    def test_get_space(self):
        space = self.dset.get_space()
        self.assertEqual(space.get_simple_extent_dims(), SHAPE)

    def test_get_space_status(self):
        status = self.dset.get_space_status()
        self.assert_(status > 0)

    # Chunked datasets have no offset.  New test dset needed.
    #
    #def test_get_offset(self):
    #    pass

    def test_get_storage_size(self):
        self.assert_(self.dset.get_storage_size() >= 0)

    def test_get_type(self):
        self.assertEqual(self.dset.get_type().dtype, DTYPE)

    def test_get_create_plist(self):
        pid = self.dset.get_create_plist()
        self.assertEqual(h5i.get_type(pid), h5i.GENPROP_LST)

    def test_py(self):
        self.assertEqual(self.dset.dtype, DTYPE)
        self.assertEqual(self.dset.shape, SHAPE)
        self.assertEqual(self.dset.rank, len(SHAPE))

