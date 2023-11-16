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

import os
import random

try:
    import blosc2 as b2
    import hdf5plugin as h5p
except ImportError:
    b2 = h5p = None
import numpy as np

from .common import ut, TestCase

from h5py import File


class StoreArrayMixin:
    # Requires: self.f (read/write), self.arr, self.chunks
    # Provides: self.f (read-only), self.dset
    def setUp(self):
        comp = h5p.Blosc2(cname='lz4', clevel=5, filters=h5p.Blosc2.SHUFFLE)
        self.f.create_dataset('x', data=self.arr, chunks=self.chunks, **comp)

        # Reopen the test file read-only to ensure
        # that no HDF5/h5py caching takes place.
        fn = self.f.filename
        self.f.close()
        self.f = File(fn, 'r')
        self.dset = self.f['x']


@ut.skipIf(b2 is None or h5p is None, 'Blosc2 support is required')
class Blosc2OptSlicingTestCase(TestCase, StoreArrayMixin):

    """
        Feature: Blosc2 optimized slicing
    """

    blosc2_force_filter = False

    def setUp(self):
        TestCase.setUp(self)

        shape = (3500, 300)
        self.chunks = (1747, 150)
        self.arr = np.arange(np.prod(shape), dtype="u2").reshape(shape)
        StoreArrayMixin.setUp(self)

        self.blosc2_filter_env = os.environ.get('BLOSC2_FILTER', '0')
        os.environ['BLOSC2_FILTER'] = '1' if self.blosc2_force_filter else '0'

    def tearDown(self):
        os.environ['BLOSC2_FILTER'] = self.blosc2_filter_env
        super().tearDown()

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

    def test_astype(self):
        """ Reading a slice converted to another type """
        alt_dtype = np.dtype('u4')
        self.assertTrue(self.dset.dtype < alt_dtype)
        alt_arr = self.arr.astype(alt_dtype)
        alt_dset = self.dset.astype(alt_dtype)
        slc = slice(10, 20)
        self.assertArrayEqual(alt_dset[slc], alt_arr[slc])


@ut.skipIf(b2 is None or h5p is None, 'Blosc2 support is required')
class Blosc2FiltSlicingTestCase(Blosc2OptSlicingTestCase):

    """
        Feature: Blosc2 filter slicing
    """

    blosc2_force_filter = True


@ut.skipIf(b2 is None or h5p is None, 'Blosc2 support is required')
class Blosc2OptSlicingMinTestCase(TestCase, StoreArrayMixin):

    """
        Feature: Blosc2 optimized slicing with chunks on inner dimension
    """

    # Minimal test which can be figured out manually::
    #
    #     z  Data: 1   Chunk0:   Chunk1: 1   Slice:
    #    /        /|\                    |\
    #   |\       0 5 3       0           5 3        5
    #   x y      |X X|       |\           \|       / \
    #            4 2 7       4 2           7      4   7
    #             \|/         \|                   \ /
    #              6           6                    6
    #
    #                  Chunk0 & Slice: 4   Chunk1 & Slice: 5
    #                                   \                   \
    #                                    6                   7
    #
    # This is mainly a test for the assemblage of the returned slice
    # from its parts in different chunks.

    def setUp(self):
        TestCase.setUp(self)

        shape = (2, 2, 2)
        self.chunks = (2, 2, 1)
        self.arr = np.arange(np.prod(shape), dtype="u1").reshape(shape)
        StoreArrayMixin.setUp(self)

    def test_slice(self):
        """ Reading a slice perpendicular to chunks """
        slc = (slice(1, 2), slice(0, 2), slice(0, 2))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])


@ut.skipIf(b2 is None or h5p is None, 'Blosc2 support is required')
class Blosc2OptSlicingCompTestCase(TestCase, StoreArrayMixin):

    """
        Feature: Blosc2 optimized slicing with compound dtypes
    """

    def setUp(self):
        TestCase.setUp(self)

        dtype = np.dtype('i4,f8')
        shape = (5, 5)
        self.chunks = (2, 2)
        arr = np.zeros(dtype=dtype, shape=shape)
        arr[0, 0] = (1, 1)
        arr[0, 2] = (2, 2)
        arr[0, 4] = (3, 3)
        arr[2, 0] = (4, 4)
        arr[2, 2] = (5, 5)
        arr[2, 4] = (6, 6)
        arr[4, 0] = (7, 7)
        arr[4, 2] = (8, 8)
        arr[4, 4] = (9, 9)
        self.arr = arr
        StoreArrayMixin.setUp(self)

    def test_whole_array(self):
        """ Reading a slice covering the whole array """
        self.assertArrayEqual(self.dset[:], self.arr)

    def test_chunk(self):
        """ Reading a slice matching a chunk """
        slcs = [tuple(slice(mult * cl, (mult + 1) * cl)
                      for cl in self.dset.chunks)
                for mult in range(3)]
        for slc in slcs:
            self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_cross_chunk(self):
        """ Reading a slice crossing chunk boundaries """
        slc = (slice(1, 1), slice(3, 3))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_last_chunk(self):
        """ Reading a slice going past the last chunk """
        slc = (slice(-1, self.dset.shape[0] + 10),
               slice(-1, self.dset.shape[1] + 10))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    def test_cross_last_chunk(self):
        """ Reading a slice crossing chunk boundaries past last chunk """
        slc = (slice(-2, self.dset.shape[0] + 10),
               slice(-2, self.dset.shape[1] + 10))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])
