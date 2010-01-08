
"""
    Tests slicing compatibility.  The following slicing schemes are supported:

    Simple slicing
        Uses any combination of integers, ":" and "...".  These translate
        to hyperslab selections and are the easiest to implement

    Broadcast slicing
        Refers to simple slicing where the shape of the selection and the
        shape of the memory data do not match.  The rules for NumPy
        broadcasting are pathologically complex.  Therefore, broadcasting is
        not supported for advanced indexing.

    Advanced indexing
        Equivalent to simple slicing, except that the following are allowed:

        1.  A list of indices, per axis
        2.  A boolean array, per axis
        3.  One large boolean array
"""



import numpy as np
import os
import unittest

from common import TestCasePlus, res

import h5py


class SliceFreezer(object):
    """ Necessary because numpy.s_ clips slices > 2**32 """
    def __getitem__(self, args):
        return args

s = SliceFreezer()

class TestSlicing(TestCasePlus):

    def setUp(self):
        self.f = h5py.File(res.get_name(), 'w')

    def tearDown(self):
        res.clear()

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

            msg = " slice %s" % (slc,)

            arr[slc] += np.random.rand()
            dset[slc] = arr[slc]
            self.assertArrayEqual(dset, arr, "write"+msg)

            out = dset[slc]
            self.assertArrayEqual(out, arr[slc], "read"+msg)

            arr[slc] += np.random.rand()
            dset.write_direct(arr, slc, slc)
            self.assertArrayEqual(dset, arr, "write direct"+msg)

            out = np.ndarray(shape, 'f')
            dset.read_direct(out, slc, slc)
            self.assertArrayEqual(out[slc], arr[slc], "read direct"+msg)

    def test_slices_big(self):
        # Test slicing behavior for indices larger than 2**32

        shape = (2**62, 2**62)
        dtype = 'f'

        bases = [1024, 2**37, 2**60]
        regions = [ (42,1), (100,100), (1,42), (1,1), (4,1025)]

        for base in bases:

            slices = [ s[base:base+x, base:base+y] for x, y in regions]

            dset = self.generate_dset(shape, dtype, maxshape=(None, None))

            for region, slc in zip(regions, slices):
                msg = "Testing base %s shape %s slice %s" % (np.log2(base), region, slc,)
        
                data = np.arange(np.product(region), dtype=dtype).reshape(region)

                dset[slc] = data

                self.assertArrayEqual(dset[slc], data, msg)

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

            dset[slc] = subarr
            arr[slc] = subarr
            self.assertArrayEqual(dset, arr, "broadcast %s %s" % (slc, shape))

    def test_scalar_broadcast(self):
        # Check scalar broadcasting for multiple types

        types = ['i', 'f', [('a', 'i'), ('b','f')]]
        values = [np.ones((), t) for t in types]

        for idx, (v, t) in enumerate(zip(values, types)):

            comparison = np.empty((100,100), dtype=t)
            comparison[...] = v

            dset = self.f.create_dataset('ds%d' % idx, (100,100), dtype=t)

            dset[...] = v

            self.assert_(np.all(dset[...] == comparison), "%d: %s %s" % (idx, v, t))

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
            msg = "slicing %s" % (slc,)
            assert np.all(dset[slc] == result), msg

    def test_fancy_index(self):
        # Test fancy selection with list indexing. Addresses I29, I31, I32.

        mydata = np.arange(4*5,dtype='i').reshape((4,5))
        dset = self.f.create_dataset("mydata", data=mydata)

        slc = s[:, [0,1,2,3,4]]
        self.assertArrayEqual(mydata[slc], dset[slc])

        self.assertRaises(TypeError, dset.__getitem__, s[0,[1,0,2]])
        self.assertRaises(TypeError, dset.__getitem__, s[[0,1], [0,1]])

        # Boolean array indexing
        barr = np.array([True, False, True, True], 'bool')
        self.assertArrayEqual(mydata[0,barr], dset[0,barr])

        # Check that NumPy arrays can be used as lists
        slc = s[:, np.array([0,1,3], dtype='i')]
        self.assertArrayEqual(mydata[slc], dset[slc])

    def test_compound_literal(self):
        # I41
        dt = np.dtype([('a', 'i'), ('b', 'f'), ('c', '|S10')])
        val = (1, 2.0, "Hello")

        dset = self.f.create_dataset("ds", (10,), dt)
        
        dset[0] = val
        
        self.assert_(np.all(dset[0]==val))



        
