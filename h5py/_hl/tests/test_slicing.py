
"""
    Dataset slicing test module.

    Tests all supported slicing operations, including read/write and
    broadcasting operations.  Does not test type conversion except for
    corner cases overlapping with slicing; for example, when selecting
    specific fields of a compound type.
"""

import numpy as np

from .common import ut, TestCase

import h5py
from h5py.highlevel import File

class BaseSlicing(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestSingleElement(BaseSlicing):

    """
        Feature: Retrieving a single element works with NumPy semantics
    """

    def test_single_index(self):
        """ Single-element selection with [index] yields array scalar """
        dset = self.f.create_dataset('x', (1,), dtype='i1')
        out = dset[0]
        self.assertIsInstance(out, np.int8)

    def test_single_null(self):
        """ Single-element selection with [()] yields ndarray """
        dset = self.f.create_dataset('x', (1,), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_scalar_index(self):
        """ Slicing with [...] yields scalar ndarray """
        dset = self.f.create_dataset('x', shape=(), dtype='f')
        out = dset[...]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, ())

    def test_scalar_null(self):
        """ Slicing with [()] yields array scalar """
        dset = self.f.create_dataset('x', shape=(), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalar is numpy.void, not tuple (issue 135) """
        dt = np.dtype([('a','i4'),('b','f8')])
        v = np.ones((4,), dtype=dt)
        dset = self.f.create_dataset('foo', (4,), data=v)
        self.assertEqual(dset[0], v[0])
        self.assertIsInstance(dset[0], np.void)

class TestObjectIndex(BaseSlicing):

    """
        Feauture: numpy.object_ subtypes map to real Python objects
    """

    def test_reference(self):
        """ Indexing a reference dataset returns a h5py.Reference instance """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.special_dtype(ref=h5py.Reference))
        dset[0] = self.f.ref
        self.assertEqual(type(dset[0]), h5py.Reference)

    def test_regref(self):
        """ Indexing a region reference dataset returns a h5py.RegionReference
        """
        dset1 = self.f.create_dataset('x', (10,10))
        regref = dset1.regionref[...]
        dset2 = self.f.create_dataset('y', (1,), dtype=h5py.special_dtype(ref=h5py.RegionReference))
        dset2[0] = regref
        self.assertEqual(type(dset2[0]), h5py.RegionReference)

    def test_reference_field(self):
        """ Compound types of which a reference is an element work right """
        reftype = h5py.special_dtype(ref=h5py.Reference)
        dt = np.dtype([('a', 'i'),('b',reftype)])

        dset = self.f.create_dataset('x', (1,), dtype=dt)
        dset[0] = (42, self.f['/'].ref)

        out = dset[0]
        self.assertEqual(type(out[1]), h5py.Reference)  # isinstance does NOT work

    def test_scalar(self):
        """ Indexing returns a real Python object on scalar datasets """
        dset = self.f.create_dataset('x', (), dtype=h5py.special_dtype(ref=h5py.Reference))
        dset[()] = self.f.ref
        self.assertEqual(type(dset[()]), h5py.Reference)

    def test_bytestr(self):
        """ Indexing a byte string dataset returns a real python byte string
        """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.special_dtype(vlen=bytes))
        dset[0] = b"Hello there!"
        self.assertEqual(type(dset[0]), bytes)

class TestSimpleSlicing(TestCase):

    """
        Feature: Simple NumPy-style slices (start:stop:step) are supported.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_negative_stop(self):
        """ Negative stop indexes work as they do in NumPy """
        self.assertArrayEqual(self.dset[2:-2], self.arr[2:-2])

class TestArraySlicing(BaseSlicing):

    """
        Feature: Array types are handled appropriately
    """

    def test_read(self):
        """ Read arrays tack array dimensions onto end of shape tuple """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x',(10,),dtype=dt)
        self.assertEqual(dset.shape, (10,))
        self.assertEqual(dset.dtype, dt)

        # Full read
        out = dset[...]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (10,3))

        # Single element
        out = dset[0]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3,))

        # Range
        out = dset[2:8:2]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3,3))

    @ut.expectedFailure
    def test_write_broadcast(self):
        """ Fill an array type with a constant 

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        dset[...] = 42.0
        out = dset[...]
        self.assertTrue(np.all(out == 42))

    @ut.expectedFailure
    def test_write_element(self):
        """ Write a single element to the array 

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)
        
        data = np.array([1,2,3.0])
        dset[4] = data
        
        out = dset[4]
        self.assertTrue(np.all(out == data))

    @ut.expectedFailure
    def test_roundtrip(self):
        """ Read the contents of an array and write them back

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)

        out = dset[...]
        dset[...] = out

        self.assertTrue(np.all(dset[...] == out))






