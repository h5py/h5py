
from tempfile import mktemp
from h5py import tests
import h5py

import numpy as np

class Base(tests.HTest):

    def setUp(self):
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        import os
        self.f.close()
        os.unlink(self.name)

class TestArray(Base):

    def test_array(self):
        """ (Dataset) Auto-conversion (2D) """
        data = np.arange(100).reshape((10,10))
        ds = self.f.create_dataset('foo', data=data)
        arr = np.array(ds)
        self.assertArrayEqual(arr, data)

    def test_scalar(self):
        """ (Dataset) Auto-conversion (scalar) """
        data = np.array(42)
        ds = self.f.create_dataset('foo', data=data)
        arr = np.array(ds)
        self.assertEqual(arr, data)

    def test_dtype(self):
        """ (Dataset) Auto-conversion (type) """
        data = np.arange(100).reshape((10,10)).astype('u8')
        ds = self.f.create_dataset('foo', data=data)
        arr = ds.__array__(np.dtype('i1'))
        self.assertEqual(arr.dtype, np.dtype('i1'))
        self.assertArrayEqual(arr, data.astype(arr.dtype))

    def test_fieldname_exc(self):
        """ (Dataset) Field name on non-compound dataset raises ValueError """
        ds = self.f.create_dataset('foo', (100,), 'f')
        self.assertRaises(ValueError, ds.__getitem__, (0, 'a'))

        
