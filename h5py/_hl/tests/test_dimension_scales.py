import sys

import numpy as np

from .common import ut, TestCase
from h5py.highlevel import File, Group, Dataset
import h5py

class BaseDataset(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCreateDimensionScale(BaseDataset):

    """
        Feature: Datasets can be created from existing data
    """

    def test_create_dimensionscale(self):
        """ Create a scalar dataset from existing array """
        data = np.ones((2), 'f')
        dset = self.f.create_dataset('foo', data=data)
        h5py.h5ds.set_scale(dset.id)
        self.assertEqual(dset.attrs['CLASS'], "DIMENSION_SCALE")


















