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

from h5py import tests
from h5py import *

class TestBasic(tests.HTest):

    def setUp(self):
        self.fid, self.name = tests.gettemp()
        self.id = h5g.open(self.fid, '/')
        sid = h5s.create_simple((10,10))
        self.did = h5d.create(self.fid, 'ds', h5t.STD_I32LE, sid)

    def tearDown(self):
        import os
        self.fid.close()
        os.unlink(self.name)

    def test_obj(self):
        """ (H5R) Object reference round-trip """
        ref = h5r.create(self.id, '/', h5r.OBJECT)
        self.assertIsInstance(ref, h5r.Reference)
        self.assertEqual(h5r.dereference(ref, self.fid), self.id)

    def test_reg(self):
        """ (H5R) Region reference round-trip """
        sid = self.did.get_space()
        ref = h5r.create(self.did, '.', h5r.DATASET_REGION, sid)
        self.assertIsInstance(ref, h5r.RegionReference)
        self.assertEqual(h5r.dereference(ref, self.fid), self.did)
        sid2 = h5r.get_region(ref, self.id)
        self.assertIsInstance(sid, h5s.SpaceID)
        self.assertEqual(sid2.get_select_bounds(), sid.get_select_bounds())

    def test_create_exc(self):
        """ (H5R) RegionReference w/no dataspace raises ValueError """
        self.assertRaises(ValueError, h5r.create, self.did, '.', h5r.DATASET_REGION)

    @tests.require(api=18)
    def test_obj(self):
        """ (H5R) get_name() on object reference """
        ref = h5r.create(self.id, '/', h5r.OBJECT)
        self.assertEqual(h5r.get_name(ref, self.fid), '/')



