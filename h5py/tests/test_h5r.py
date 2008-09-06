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
import tempfile
import os
import numpy

from h5py import *
from h5py.h5 import H5Error


class TestH5R(unittest.TestCase):

    def setUp(self):
        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')
        self.dset = self.f.create_dataset("ds", (100,))
        self.grp = self.f.create_group("grp")

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)

    def test_objref(self):
        ref = h5r.create(self.f.id, "grp", h5r.OBJECT)
        deref = ref.dereference(self.f.id)
        self.assertEqual(deref, self.grp.id)
        #self.assertEqual(h5i.get_name(deref), h5i.get_name(self.grp.id))
        self.assertEqual(ref.get_obj_type(self.f.id), h5g.GROUP)

    def test_regref(self):
        space = self.dset.id.get_space()
        space.select_hyperslab((13,), (42,))
        ref = h5r.create(self.f.id, "ds", h5r.DATASET_REGION, space)

        #deref = ref.dereference(self.f.id)
        #self.assertEqual(deref, self.grp.id)
        #self.assertEqual(h5i.get_name(deref), h5i.get_name(self.grp.id))      

        deref_space = ref.get_region(self.f.id)
        self.assertEqual(space.shape, deref_space.shape)
        self.assert_(space.get_select_bounds(), deref_space.get_select_bounds())

        self.assertEqual(ref.get_obj_type(self.f.id), h5g.DATASET)





