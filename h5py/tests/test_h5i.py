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

import h5py
from h5py import h5f, h5g, h5i, h5t
from h5py.errors import H5Error

HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/attributes.hdf5')
OBJECTNAME = 'Group'

class TestH5I(unittest.TestCase):

    def setUp(self):
        self.fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        h5g.close(self.obj)
        h5f.close(self.fid)

    def test_get_type(self):
        self.assertEqual(h5i.get_type(self.fid), h5i.TYPE_FILE)
        self.assertEqual(h5i.get_type(self.obj), h5i.TYPE_GROUP)
        self.assertEqual(h5i.get_type(-1), h5i.TYPE_BADID)

    def test_get_name(self):
        self.assertEqual(h5i.get_name(self.obj), '/Group')
        self.assertEqual(h5i.get_name(h5t.STD_I8LE), None)
        self.assertEqual(h5i.get_name(-1), None)










