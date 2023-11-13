# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2023 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Dataset Blosc2 optimized slicing test module.

    Tests slice read operations for the cases where Blosc2 optimized slicing
    can be used.
"""

import random

try:
    import blosc2 as b2
    import hdf5plugin as h5p
except ImportError:
    b2 = h5p = None
import numpy as np

from .common import ut, TestCase

from h5py import File

@ut.skipIf(b2 is None or h5p is None, 'Blosc2 support is required')
class Blosc2OptSlicingTestCase(TestCase):

    """
        Feature: Blosc2 optimized slicing
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        shape = (3500, 300)
        chunks = (1747, 150)
        comp = h5p.Blosc2(cname='lz4', clevel=5, filters=h5p.Blosc2.SHUFFLE)
        self.arr = np.arange(np.prod(shape), dtype="u2").reshape(shape)
        self.dset = self.f.create_dataset('x', data=self.arr, chunks=chunks,
                                          **comp)

    def tearDown(self):
        if self.f:
            self.f.close()

    # Test the data of the returned object.

    def test_whole_array(self):
        """ Reading a slice covering the whole array """
        self.assertArrayEqual(self.dset[:], self.arr)

    def test_cross_chunk_1dim(self):
        """ Reading a slice crossing chunk boundaries (1-dim) """
        slc = slice(self.dset.chunks[0] - 5, self.dset.chunks[0] + 5)
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_cross_chunk_ndim(self):
        """ Reading a slice crossing chunk boundaries (n-dim) """
        slc = (slice(self.dset.chunks[0] - 5, self.dset.chunks[0] + 5),
               slice(self.dset.chunks[1] - 5, self.dset.chunks[1] + 5))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_last_chunk_1dim(self):
        """ Reading a slice going past the last chunk (1-dim) """
        slc = slice(self.dset.shape[0] - 5, self.dset.shape[0] + 5)
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_last_chunk_ndim(self):
        """ Reading a slice going past the last chunk (n-dim) """
        slc = (slice(self.dset.shape[0] - 5, self.dset.shape[0] + 5),
               slice(self.dset.shape[1] - 5, self.dset.shape[1] + 5))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    # Test the attributes of the returned object.

    def test_scalar_inside(self):
        """ Reading a scalar inside of the array """
        coord = tuple(random.randrange(0, c) for c in self.dset.shape)
        self.assertEqual(self.dset[coord], self.arr[coord])

    def test_scalar_outside(self):
        """ Reading a scalar outside of the array """
        shape = self.dset.shape
        coords = [(shape[0] * 2, 0), (0, shape[1] * 2),
                  tuple(c * 2 for c in shape)]
        for coord in coords:
            with self.assertRaises(IndexError):
                self.dset[coord]

    def test_slice_outside(self):
        """ Reading a slice outside of the array (empty) """
        shape = self.dset.shape
        slcs = [(slice(shape[0] * 2, shape[0] * 3), ...),
                (..., slice(shape[1] * 2, shape[1] * 3)),
                tuple(slice(c * 2, c * 3) for c in shape)]
        for slc in slcs:
            self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_slice_1dimless(self):
        """ Reading a slice with one dimension less than the array """
        idxs = [random.randrange(0, dim) for dim in self.dset.shape]
        for idx in idxs:
            self.assertArrayEqual(self.dset[idx], self.arr[idx])
