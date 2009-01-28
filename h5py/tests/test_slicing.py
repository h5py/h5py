import numpy as np
import os
from nose.tools import assert_equal

from common import makehdf, delhdf

import h5py

def check_arr_equal(dset, arr):
    """ Make sure dset and arr have the same shape, dtype and contents.

        Note that dset may be a NumPy array or an HDF5 dataset
    """
    if np.isscalar(dset) or np.isscalar(arr):
        assert np.isscalar(dset) and np.isscalar(arr)
        assert dset == arr
        return

    assert_equal(dset.shape, arr.shape)
    assert_equal(dset.dtype, arr.dtype)
    assert np.all(dset[...] == arr[...]), "%s %s" % (dset[...], arr[...])

class SliceFreezer(object):
    """ Necessary because numpy.s_ clips slices > 2**32 """
    def __getitem__(self, args):
        return args

s = SliceFreezer()

class TestSlicing(object):

    def setUp(self):
        self.f = makehdf()

    def tearDown(self):
        delhdf(self.f)

    def generate(self, shape, dtype):
        if 'dset' in self.f:
            del self.f['dset']

        size = np.product(shape)
        dset = self.f.create_dataset('dset', shape, dtype)
        arr = np.arange(size, dtype=dtype).reshape(shape)
        return dset, arr

    def generate_rand(self, shape, dtype='f'):
        return np.random.random(shape).astype(dtype)


    def test_slices(self):
        dset, arr = self.generate((10,10,50),'f')

        slices = [s[0,0,0], s[0,0,:], s[0,:,0], s[0,:,:]]
        slices += [s[0:1,:,4:5], s[2:3,0,4:5], s[:,0,0:1], s[0,:,0:1]]
        slices += [ s[9,9,49], s[9,:,49], s[9,:,:] ]
        slices += [ s[0, ..., 49], s[...], s[..., 49], s[9,...] ]
        slices += [ s[0:7:2,0:9:3,15:43:5], s[2:8:2,...] ]
        slices += [ s[0], s[1], s[9], s[0,0], s[4,5], s[:] ]
        slices += [ s[3,...], s[3,2,...] ]
        slices += [ np.random.random((10,10,50)) > 0.5 ]  # Truth array
        slices += [ np.zeros((10,10,50), dtype='bool') ]
        slices += [ s[0, 1, [2,3,6,7]], s[:,[1,2]], s[[1,2]], s[3:7,[1]]]
        
        dset[...] = arr[...]

        for slc in slices:

            arr[slc] += np.random.rand()
            dset[slc] = arr[slc]
            
            print "check write %s" % (slc,)
            check_arr_equal(dset, arr)

            out = dset[slc]

            print "check read %s" % (slc,)
            check_arr_equal(out, arr[slc])


    def test_scalars(self):

        dset, arr = self.generate((),'i')
        dset[...] = arr[...] = 42
        assert dset[...] == dset[()] == arr[()] == 42
        assert dset.shape == ()
        assert np.isscalar(dset[...])


    def test_broadcast(self):

        dset, arr = self.generate((20,10,30),'f')
        dset[...] = arr[...]

        slices = [(s[...], (30,)),
                  (s[...], (10,30)),
                  (s[:,5,:], (20,30)),
                  (s[:,4,:], (30,)),
                  (s[:,3,...], (1,30,))]

        for slc, shape in slices:

            subarr = np.random.random(shape)

            print "broadcast %s %s" % (slc, shape)
            dset[slc] = subarr
            arr[slc] = subarr
            check_arr_equal(dset, arr)






        
