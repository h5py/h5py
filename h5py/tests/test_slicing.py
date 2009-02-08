import numpy as np
import os
from nose.tools import assert_equal

from common import makehdf, delhdf, assert_arr_equal, skip

import h5py


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

        size = np.product(shape)
        dset = self.generate_dset(shape, dtype)
        arr = np.arange(size, dtype=dtype).reshape(shape)
        return dset, arr

    def generate_dset(self, shape, dtype, **kwds):
        if 'dset' in self.f:
            del self.f['dset']
        return self.f.create_dataset('dset', shape, dtype, **kwds)
        
    def generate_rand(self, shape, dtype='f'):
        return np.random.random(shape).astype(dtype)


    def test_slices(self):
        # Test interger, slice, array and list indices

        shape = (10,10,50)
        dset, arr = self.generate(shape,'f')

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

            print "slice %s" % (slc,)

            print "    write"
            arr[slc] += np.random.rand()
            dset[slc] = arr[slc]
            assert_arr_equal(dset, arr)

            print "    read"
            out = dset[slc]
            assert_arr_equal(out, arr[slc])

            print "    write direct"
            arr[slc] += np.random.rand()
            dset.write_direct(arr, slc, slc)
            assert_arr_equal(dset, arr)

            print "    read direct"
            out = np.ndarray(shape, 'f')
            dset.read_direct(out, slc, slc)
            assert_arr_equal(out[slc], arr[slc])

    def test_slices_big(self):
        # Test slicing behavior for indices larger than 2**32

        shape = (2**62, 2**62)
        dtype = 'f'

        bases = [1024, 2**37, 2**60]
        regions = [ (42,1), (100,100), (1,42), (1,1), (4,1025)]

        for base in bases:
            print "Testing base 2**%d" % np.log2(base)

            slices = [ s[base:base+x, base:base+y] for x, y in regions]

            dset = self.generate_dset(shape, dtype, maxshape=(None, None))

            for region, slc in zip(regions, slices):
                print "    Testing shape %s slice %s" % (region, slc,)
        
                data = np.arange(np.product(region), dtype=dtype).reshape(region)

                dset[slc] = data

                assert_arr_equal(dset[slc], data)

    def test_scalars(self):
        # Confirm correct behavior for scalar datasets

        dset, arr = self.generate((),'i')
        dset[...] = arr[...] = 42
        assert dset[...] == dset[()] == arr[()] == 42
        assert dset.shape == ()
        assert np.isscalar(dset[...])


    def test_broadcast(self):
        # Test broadcasting to HDF5

        dset, arr = self.generate((20,10,30),'f')
        dset[...] = arr[...]

        slices = [(s[...], ()),
                  (s[...], (30,)),
                  (s[...], (10,30)),
                  (s[:,5,:], (20,30)),
                  (s[:,4,:], (30,)),
                  (s[:,3,...], (1,30,))]

        for slc, shape in slices:

            subarr = np.random.random(shape)

            print "broadcast %s %s" % (slc, shape)
            dset[slc] = subarr
            arr[slc] = subarr
            assert_arr_equal(dset, arr)


    @skip
    def test_broadcast_big(self):

        M = 1024*1024

        dset = self.f.create_dataset('dset', (100,0.5*M), 'i')

        dset[...] = 42

        comprow = np.ones((0.5*M,),dtype='i')*42

        for row in dset:
            assert np.all(row == comprow)

    def test_slice_names(self):
        # Test slicing in conjunction with named fields

        shape = (10,10)
        size = np.product(shape)
        dtype = [('a', 'i'), ('b', 'f')]

        srcarr = np.ndarray(shape, dtype)

        srcarr['a'] = np.arange(size).reshape(shape)
        srcarr['b'] = np.arange(size).reshape(shape)*100

        dset = self.f.create_dataset('TEST', data=srcarr)

        pairs = [  (s[:], srcarr[:]),
                   (s['a'], srcarr['a']),
                   (s[5,5,'a'], srcarr['a'][5,5]),
                   (s[2,:,'b'], srcarr['b'][2,:]),
                   (s['b',...,5], srcarr[...,5]['b']) ]

        for slc, result in pairs:
            print "slicing %s" % (slc,)
            assert np.all(dset[slc] == result)




        
