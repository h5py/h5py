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

import numpy
from common import TestCasePlus

from h5py import *

HDFNAME = 'smpl_compound_chunked.hdf5'
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

class TestH5D(TestCasePlus):


    def setUp(self):
        self.setup_fid(HDFNAME)
        self.dset = h5d.open(self.fid, "CompoundChunked")

    def tearDown(self):
        self.dset._close()
        self.teardown_fid()

    def test_open_close(self):
        dset = h5d.open(self.fid, "CompoundChunked")
        self.assertEqual(h5i.get_type(dset), h5i.DATASET)
        dset._close()
        self.assertEqual(h5i.get_type(dset), h5i.BADID)

    def test_read(self):
        array = numpy.ndarray(SHAPE, dtype=DTYPE)

        self.dset.read(h5s.ALL, h5s.ALL, array)
        for name in DTYPE.fields:
            self.assert_(numpy.all(array[name] == basearray[name]), "%s::\n%s\n!=\n%s" % (name, array[name], basearray[name]))

    def test_write(self):
        root = h5g.open(self.fid, '/')
        dt = numpy.dtype('<i4')
        space = h5s.create_simple((10,20))
        htype = h5t.py_create(dt)
        arr = 5*numpy.ones((10,20),dtype=dt)
        dset = h5d.create(root, 'NewDataset', htype, space)
        dset.write(h5s.ALL, h5s.ALL, arr)
        arr2 = numpy.ndarray((10,20), dtype=dt)
        dset.read(h5s.ALL, h5s.ALL, arr2)
        self.assert_((arr == arr2).all())

    def test_get_space(self):
        space = self.dset.get_space()
        self.assertEqual(space.get_simple_extent_dims(), SHAPE)

    def test_get_space_status(self):
        status = self.dset.get_space_status()
        self.assert_(status > 0)

    def test_create_offset(self):
        root = h5g.open(self.fid, '/')
        space = h5s.create_simple((10,20))
        htype = h5t.STD_I32LE
        dset = h5d.create(root, 'NewDataset', htype, space)
        dset = h5d.open(root, 'NewDataset')
        self.assertEqual(dset.dtype, htype.dtype)
        self.assertEqual(dset.shape, space.shape)
        dset.get_offset()
        self.dset.get_offset()

    def test_extend(self):
        plist = h5p.create(h5p.DATASET_CREATE)
        plist.set_chunk((1,20))
        root = h5g.open(self.fid, '/')
        space = h5s.create_simple((10,20),(15,20))
        htype = h5t.STD_I32LE
        dset = h5d.create(root, 'NewDataset', htype, space, dcpl=plist)
        self.assertEqual(dset.shape, (10,20))
        dset.extend((15,20))
        self.assertEqual(dset.shape, (15,20))

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

