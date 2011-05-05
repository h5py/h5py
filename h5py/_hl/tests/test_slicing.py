import numpy as np

from .common import ut, TestCase

from h5py.highlevel import File

class BaseSlicing(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCompound(BaseSlicing):

    """
        Feature: Slicing works with compound types
    """

    def test_scalar_type(self):
        """ Structure scalar is numpy.void, not tuple (issue 135) """
        dt = np.dtype([('a','i4'),('b','f8')])
        v = np.ones((4,), dtype=dt)
        dset = self.f.create_dataset('foo', (4,), data=v)
        self.assertEqual(dset[0], v[0])
        self.assertIsInstance(dset[0], np.void)

