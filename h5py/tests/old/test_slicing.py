# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Dataset slicing test module.

    Tests all supported slicing operations, including read/write and
    broadcasting operations.  Does not test type conversion except for
    corner cases overlapping with slicing; for example, when selecting
    specific fields of a compound type.
"""

from __future__ import absolute_import

import six

import numpy as np

from ..common import ut, TestCase

import h5py
from h5py import h5s, h5t, h5d
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

    def test_write_broadcast(self):
        """ Array fill from constant is not supported (issue 211).
        """
        dt = np.dtype('(3,)i')

        dset = self.f.create_dataset('x', (10,), dtype=dt)

        with self.assertRaises(TypeError):
            dset[...] = 42

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

    def test_write_slices(self):
        """ Write slices to array type """
        dt = np.dtype('(3,)i')

        data1 = np.ones((2,), dtype=dt)
        data2 = np.ones((4,5), dtype=dt)

        dset = self.f.create_dataset('x', (10,9,11), dtype=dt)

        dset[0,0,2:4] = data1
        self.assertArrayEqual(dset[0,0,2:4], data1)

        dset[3, 1:5, 6:11] = data2
        self.assertArrayEqual(dset[3, 1:5, 6:11], data2)


    def test_roundtrip(self):
        """ Read the contents of an array and write them back

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)

        out = dset[...]
        dset[...] = out

        self.assertTrue(np.all(dset[...] == out))


class TestZeroLengthSlicing(BaseSlicing):

    """
        Slices resulting in empty arrays
    """

    def test_slice_zero_length_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along the zero-length dimension """
        for i, shape in enumerate([(0,), (0, 3), (0, 2, 1)]):
            dset = self.f.create_dataset('x%d'%i, shape, dtype=np.int, maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[...]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            out = dset[:]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            if len(shape) > 1:
                out = dset[:, :1]
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape[:2], (0, 1))

    def test_slice_other_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along a non-zero-length dimension """
        for i, shape in enumerate([(3, 0), (1, 2, 0), (2, 0, 1)]):
            dset = self.f.create_dataset('x%d'%i, shape, dtype=np.int, maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (1,)+shape[1:])

    def test_slice_of_length_zero(self):
        """ Get a slice of length zero from a non-empty dataset """
        for i, shape in enumerate([(3,), (2, 2,), (2,  1, 5)]):
            dset = self.f.create_dataset('x%d'%i, data=np.zeros(shape, np.int), maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[1:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (0,)+shape[1:])

class TestFieldNames(BaseSlicing):

    """
        Field names for read & write
    """

    dt = np.dtype([('a', 'f'), ('b', 'i'), ('c', 'f4')])
    data = np.ones((100,), dtype=dt)

    def setUp(self):
        BaseSlicing.setUp(self)
        self.dset = self.f.create_dataset('x', (100,), dtype=self.dt)
        self.dset[...] = self.data

    def test_read(self):
        """ Test read with field selections (bytes and unicode) """
        if six.PY2:
            # Byte strings are only allowed for field names on Py2
            self.assertArrayEqual(self.dset[b'a'], self.data['a'])
        self.assertArrayEqual(self.dset[u'a'], self.data['a'])

    def test_unicode_names(self):
        """ Unicode field names for for read and write """
        self.assertArrayEqual(self.dset[u'a'], self.data['a'])
        self.dset[u'a'] = 42
        data = self.data.copy()
        data['a'] = 42
        self.assertArrayEqual(self.dset[u'a'], data['a'])

    def test_write(self):
        """ Test write with field selections """
        data2 = self.data.copy()
        data2['a'] *= 2
        self.dset['a'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['b'] *= 4
        self.dset['b'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['a'] *= 3
        data2['c'] *= 3
        self.dset['a','c'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))

    def test_write_noncompound(self):
        """ Test write with non-compound source (single-field) """
        data2 = self.data.copy()
        data2['b'] = 1.0
        self.dset['b'] = 1.0
        self.assertTrue(np.all(self.dset[...] == data2))
