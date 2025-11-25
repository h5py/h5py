# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.AttributeManager.create() method.
"""

import numpy as np
from .. import h5t, h5a

from .common import TestCase, make_name

class TestArray(TestCase):

    """
        Check that top-level array types can be created and read.
    """

    def test_int(self):
        # See issue 498
        name = make_name()
        dt = np.dtype('(3,)i')
        data = np.arange(3, dtype='i')

        self.f.attrs.create(name, data=data, dtype=dt)

        aid = h5a.open(self.f.id, name.encode('utf-8'))

        htype = aid.get_type()
        self.assertEqual(htype.get_class(), h5t.ARRAY)

        out = self.f.attrs[name]

        self.assertArrayEqual(out, data)

    def test_string_dtype(self):
        # See issue 498 discussion

        self.f.attrs.create(make_name(), data=42, dtype='i8')

    def test_str(self):
        # See issue 1057
        name = make_name()
        self.f.attrs.create(name, chr(0x03A9))
        out = self.f.attrs[name]
        self.assertEqual(out, chr(0x03A9))
        self.assertIsInstance(out, str)

    def test_tuple_of_unicode(self):
        # Test that a tuple of unicode strings can be set as an attribute. It will
        # be converted to a numpy array of vlen unicode type:
        name = make_name()
        data = ('a', 'b')
        self.f.attrs.create(name, data=data)
        result = self.f.attrs[name]
        self.assertTrue(all(result == data))
        self.assertEqual(result.dtype, np.dtype('O'))

    def test_unicode_np_array(self):
        # However, a numpy array of type U being passed in will not be
        # automatically converted, and should raise an error as it does
        # not map to a h5py dtype
        data = np.array(['a', 'b'], dtype='U1')
        with self.assertRaises(TypeError):
            self.f.attrs.create('x', data=data)

    def test_shape_scalar(self):
        name = make_name()
        self.f.attrs.create(name, data=42, shape=1)
        result = self.f.attrs[name]
        self.assertEqual(result.shape, (1,))

    def test_shape_array(self):
        name = make_name()
        self.f.attrs.create(name, data=np.arange(3), shape=3)
        result = self.f.attrs[name]
        self.assertEqual(result.shape, (3,))

    def test_dtype(self):
        dt = np.dtype('(3,)i')
        array = np.arange(3, dtype='i')
        self.f.attrs.create(make_name(), data=array, dtype=dt)
        # Array dtype shape is incompatible with data shape
        array = np.arange(4, dtype='i')
        with self.assertRaises(ValueError):
            self.f.attrs.create('x', data=array, dtype=dt)
        # Shape of new attribute conflicts with shape of data
        dt = np.dtype('()i')
        with self.assertRaises(ValueError):
            self.f.attrs.create('x', data=array, shape=(5,), dtype=dt)

    def test_key_type(self):
        with self.assertRaises(TypeError):
            self.f.attrs.create(1, data=('a', 'b'))
