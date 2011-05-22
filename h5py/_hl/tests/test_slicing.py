import numpy as np

from .common import ut, TestCase

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




