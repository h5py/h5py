# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Attribute data transfer testing module

    Covers all data read/write and type-conversion operations for attributes.
"""

import numpy as np

from .common import TestCase, make_name

import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace


class BaseAttrs(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestScalar(BaseAttrs):

    """
        Feature: Scalar types map correctly to array scalars
    """

    def test_int(self):
        """ Integers are read as correct NumPy type """
        name = make_name()
        self.f.attrs[name] = np.array(1, dtype=np.int8)
        out = self.f.attrs[name]
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalars are read as numpy.void """
        name = make_name()
        dt = np.dtype([('a', 'i'), ('b', 'f')])
        data = np.array((1, 4.2), dtype=dt)
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertIsInstance(out, np.void)
        self.assertEqual(out, data)
        self.assertEqual(out['b'], data['b'])

    def test_compound_with_vlen_fields(self):
        """ Compound scalars with vlen fields can be written and read """
        name = make_name()
        dt = np.dtype([('a', h5py.vlen_dtype(np.int32)),
                       ('b', h5py.vlen_dtype(np.int32))])

        data = np.array((np.array(list(range(1, 5)), dtype=np.int32),
                        np.array(list(range(8, 10)), dtype=np.int32)), dtype=dt)[()]

        self.f.attrs[name] = data
        out = self.f.attrs[name]

        # Specifying check_alignment=False because vlen fields have 8 bytes of padding
        # because the vlen datatype in hdf5 occupies 16 bytes
        self.assertArrayEqual(out, data, check_alignment=False)

    def test_nesting_compound_with_vlen_fields(self):
        """ Compound scalars with nested compound vlen fields can be written and read """
        dt_inner = np.dtype([('a', h5py.vlen_dtype(np.int32)),
                             ('b', h5py.vlen_dtype(np.int32))])

        dt = np.dtype([('f1', h5py.vlen_dtype(dt_inner)),
                       ('f2', np.int64)])

        inner1 = (np.array(range(1, 3), dtype=np.int32),
                  np.array(range(6, 9), dtype=np.int32))

        inner2 = (np.array(range(10, 14), dtype=np.int32),
                  np.array(range(16, 20), dtype=np.int32))

        data = np.array((np.array([inner1, inner2], dtype=dt_inner),
                         2),
                        dtype=dt)[()]

        name = make_name()
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertArrayEqual(out, data, check_alignment=False)

    def test_vlen_compound_with_vlen_string(self):
        """ Compound scalars with vlen compounds containing vlen strings can be written and read """
        dt_inner = np.dtype([('a', h5py.string_dtype()),
                             ('b', h5py.string_dtype())])

        dt = np.dtype([('f', h5py.vlen_dtype(dt_inner))])

        name = make_name()
        data = np.array((np.array([(b"apples", b"bananas"), (b"peaches", b"oranges")], dtype=dt_inner),),dtype=dt)[()]
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertArrayEqual(out, data, check_alignment=False)


class TestArray(BaseAttrs):

    """
        Feature: Non-scalar types are correctly retrieved as ndarrays
    """

    def test_single(self):
        """ Single-element arrays are correctly recovered """
        name = make_name()
        data = np.ndarray((1,), dtype='f')
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_multi(self):
        """ Rank-1 arrays are correctly recovered """
        name = make_name()
        data = np.ndarray((42,), dtype='f')
        data[:] = 42.0
        data[10:35] = -47.0
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (42,))
        self.assertArrayEqual(out, data)


class TestTypes(BaseAttrs):

    """
        Feature: All supported types can be stored in attributes
    """

    def test_int(self):
        """ Storage of integer types """
        name = make_name()
        dtypes = (np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42
            self.f.attrs[name] = data
            out = self.f.attrs[name]
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_float(self):
        """ Storage of floating point types """
        name = make_name()
        dtypes = tuple(np.dtype(x) for x in ('<f4', '>f4', '>f8', '<f8'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42.3
            self.f.attrs[name] = data
            out = self.f.attrs[name]
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_complex(self):
        """ Storage of complex types """
        name = make_name()
        dtypes = tuple(np.dtype(x) for x in ('<c8', '>c8', '<c16', '>c16'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = -4.2j + 35.9
            self.f.attrs[name] = data
            out = self.f.attrs[name]
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_string(self):
        """ Storage of fixed-length strings """
        name = make_name()
        dtypes = tuple(np.dtype(x) for x in ('|S1', '|S10'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 'h'
            self.f.attrs[name] = data
            out = self.f.attrs[name]
            self.assertEqual(out.dtype, dt)
            self.assertEqual(out[0], data[0])

    def test_bool(self):
        """ Storage of NumPy booleans """
        name = make_name()
        data = np.ndarray((2,), dtype=np.bool_)
        data[...] = True, False
        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertEqual(out.dtype, data.dtype)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_vlen_string_array(self):
        """ Storage of vlen byte string arrays"""
        name = make_name()
        dt = h5py.string_dtype(encoding='ascii')

        data = np.ndarray((2,), dtype=dt)
        data[...] = "Hello", "Hi there!  This is HDF5!"

        self.f.attrs[name] = data
        out = self.f.attrs[name]
        self.assertEqual(out.dtype, dt)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_string_scalar(self):
        """ Storage of variable-length byte string scalars (auto-creation) """
        name = make_name()
        self.f.attrs[name] = b'Hello'
        out = self.f.attrs[name]

        self.assertEqual(out, 'Hello')
        self.assertEqual(type(out), str)

        aid = h5py.h5a.open(self.f.id, name.encode('utf-8'))
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
        self.assertTrue(tid.is_variable_str())

    def test_unicode_scalar(self):
        """ Storage of variable-length unicode strings (auto-creation) """
        name = make_name()
        self.f.attrs[name] = u"Hello" + chr(0x2340) + u"!!"
        out = self.f.attrs[name]
        self.assertEqual(out, u"Hello" + chr(0x2340) + u"!!")
        self.assertEqual(type(out), str)

        aid = h5py.h5a.open(self.f.id, name.encode('utf-8'))
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)
        self.assertTrue(tid.is_variable_str())


class TestEmpty(BaseAttrs):

    def setUp(self):
        BaseAttrs.setUp(self)
        sid = h5s.create(h5s.NULL)
        tid = h5t.C_S1.copy()
        tid.set_size(10)
        aid = h5a.create(self.f.id, b'x', tid, sid)
        self.empty_obj = h5py.Empty(np.dtype("S10"))

    def test_read(self):
        self.assertEqual(
            self.empty_obj, self.f.attrs['x']
        )

    def test_write(self):
        name = make_name()
        self.f.attrs[name] = self.empty_obj
        self.assertTrue(
            is_empty_dataspace(h5a.open(self.f.id, name.encode("utf-8")))
        )

    def test_modify(self):
        with self.assertRaises(OSError):
            self.f.attrs.modify('x', 1)

    def test_values(self):
        # list() is for Py3 where these are iterators
        values = list(self.f.attrs.values())
        self.assertEqual(
            [self.empty_obj], values
        )

    def test_items(self):
        items = list(self.f.attrs.items())
        self.assertEqual(
            [(u"x", self.empty_obj)], items
        )

    def test_itervalues(self):
        values = list(self.f.attrs.values())
        self.assertEqual(
            [self.empty_obj], values
        )

    def test_iteritems(self):
        items = list(self.f.attrs.items())
        self.assertEqual(
            [(u"x", self.empty_obj)], items
        )


class TestWriteException(BaseAttrs):

    """
        Ensure failed attribute writes don't leave garbage behind.
    """

    def test_write(self):
        """ ValueError on string write wipes out attribute """

        s = b"Hello\x00Hello"

        with self.assertRaises(ValueError):
            self.f.attrs["x"] = s
        with self.assertRaises(KeyError):
            self.f.attrs["x"]
