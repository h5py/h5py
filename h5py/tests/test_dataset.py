
"""

    Test basic behavior of h5py.highlevel.Dataset, not including slicing
    or keyword arguments
"""

import numpy as np

import h5py
import unittest
from common import makehdf, delhdf, assert_arr_equal,\
                   INTS, FLOATS, COMPLEX, STRINGS, res

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.f = h5py.File(res.get_name(), 'w')
    
    def tearDown(self):
        res.clear()

    def make_dset(self, *args, **kwds):
        if 'dset' in self.f:
            del self.f['dset']
        return self.f.create_dataset('dset', *args, **kwds)

    def test_create(self):
        # Test dataset creation from shape and type, or raw data

        types = INTS + FLOATS + COMPLEX + STRINGS + ('b',)
        shapes = ( (), (1,), (10,), (20,1,15), (7,200,1) )

        for s in shapes:
            srcdata = np.arange(np.product(s)).reshape(s)

            for t in types:
                msg = "test %s %s" % (s, t)
                data = srcdata.astype(t)

                dset = self.make_dset(s, t)

                dset[...] = data

                assert np.all(dset[...] == data), "%r %r" % (dset[...], data)

                dset = self.make_dset(data=data)
 
                assert np.all(dset[...] == data), msg

                
                
