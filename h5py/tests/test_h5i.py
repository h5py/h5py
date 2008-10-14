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

from common import HDF5TestCase

from h5py import *
from h5py.h5 import H5Error

HDFNAME = 'attributes.hdf5'
OBJECTNAME = 'Group'

class TestH5I(HDF5TestCase):


    def setUp(self):
        self.setup_fid(HDFNAME)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        self.obj._close()
        self.teardown_fid()

    def test_get_type(self):
        self.assertEqual(h5i.get_type(self.fid), h5i.FILE)
        self.assertEqual(h5i.get_type(self.obj), h5i.GROUP)

    def test_get_name(self):
        self.assertEqual(h5i.get_name(self.obj), '/Group')
        self.assertEqual(h5i.get_name(h5t.STD_I8LE), None)

    def test_get_file_id(self):
        nfid = h5i.get_file_id(self.obj)
        self.assertEqual(nfid, self.fid)

    def test_refs(self):
        refcnt = h5i.get_ref(self.obj)
        self.assert_(refcnt >= 0)
        
        h5i.inc_ref(self.obj)
        self.assertEqual(h5i.get_ref(self.obj), refcnt+1)

        h5i.dec_ref(self.obj)
        self.assertEqual(h5i.get_ref(self.obj), refcnt)







