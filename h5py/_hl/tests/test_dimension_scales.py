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
        """ Create a dimension scale from existing dataset """
        data = np.ones((2), 'f')
        dset = self.f.create_dataset('foo', data=data)
        h5py.h5ds.set_scale(dset.id)
        self.assertTrue(h5py.h5ds.is_scale(dset.id))
        self.assertEqual(h5py.h5ds.get_scale_name(dset.id), '')
        self.assertEqual(dset.attrs['CLASS'], "DIMENSION_SCALE")

        dset = self.f.create_dataset('bar', data=data)
        h5py.h5ds.set_scale(dset.id, 'bar name')
        self.assertEqual(h5py.h5ds.get_scale_name(dset.id), 'bar name')

    def test_attach_dimensionscale(self):
        data = np.ones((2, 2), 'f')
        dset = self.f.create_dataset('foo', data=data)
        dscale = np.ones((2), 'f')
        dsetscale = self.f.create_dataset('bar', data=dscale)
        h5py.h5ds.set_scale(dsetscale.id)
        h5py.h5ds.attach_scale(dset.id, dsetscale.id, 0)
        self.assertTrue(h5py.h5ds.is_attached(dset.id, dsetscale.id, 0))
        self.assertFalse(h5py.h5ds.is_attached(dset.id, dsetscale.id, 1))
        self.assertEqual(h5py.h5ds.get_num_scales(dset.id, 0), 1)
        self.assertEqual(h5py.h5ds.get_num_scales(dset.id, 1), 0)
        self.assertEqual(dset.attrs.keys(), ['DIMENSION_LIST'])

    def test_detach_dimensionscale(self):
        data = np.ones((2, 2), 'f')
        dset = self.f.create_dataset('foo', data=data)
        dscale = np.ones((2), 'f')
        dsetscale = self.f.create_dataset('bar', data=dscale)
        h5py.h5ds.set_scale(dsetscale.id)
        h5py.h5ds.attach_scale(dset.id, dsetscale.id, 0)
        self.assertTrue(h5py.h5ds.is_attached(dset.id, dsetscale.id, 0))
        h5py.h5ds.detach_scale(dset.id, dsetscale.id, 0)
        self.f.flush()
        self.assertFalse(h5py.h5ds.is_attached(dset.id, dsetscale.id, 0))
        self.assertEqual(h5py.h5ds.get_num_scales(dset.id, 0), 0)

    def test_label_dimensionscale(self):
        data = np.ones((2, 2), 'f')
        dset = self.f.create_dataset('foo', data=data)
        dscale = np.ones((2), 'f')
        dsetscale = self.f.create_dataset('bar', data=dscale)
        h5py.h5ds.set_scale(dsetscale.id)
        h5py.h5ds.attach_scale(dset.id, dsetscale.id, 0)
        h5py.h5ds.set_label(dset.id, 0, 'foo label')
        h5py.h5ds.set_label(dset.id, 1, 'bar label')
        self.assertEqual(h5py.h5ds.get_label(dset.id, 0), 'foo label')
        self.assertEqual(h5py.h5ds.get_label(dset.id, 1), 'bar label')




