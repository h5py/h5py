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

import numpy as np
import pytest

from .common import TestCase, make_name

import h5py
from h5py import File, MultiBlockSlice
from h5py._hl import dataset as dataset_mod

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
        dset = self.f.create_dataset(make_name(), (1,), dtype='i1')
        out = dset[0]
        self.assertIsInstance(out, np.int8)

    def test_single_null(self):
        """ Single-element selection with [()] yields ndarray """
        dset = self.f.create_dataset(make_name(), (1,), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_scalar_index(self):
        """ Slicing with [...] yields scalar ndarray """
        dset = self.f.create_dataset(make_name(), shape=(), dtype='f')
        out = dset[...]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, ())

    def test_scalar_null(self):
        """ Slicing with [()] yields array scalar """
        dset = self.f.create_dataset(make_name(), shape=(), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalar is numpy.void, not tuple (issue 135) """
        dt = np.dtype([('a','i4'),('b','f8')])
        v = np.ones((4,), dtype=dt)
        dset = self.f.create_dataset(make_name(), (4,), data=v)
        self.assertEqual(dset[0], v[0])
        self.assertIsInstance(dset[0], np.void)

class TestObjectIndex(BaseSlicing):

    """
        Feature: numpy.object_ subtypes map to real Python objects
    """

    def test_reference(self):
        """ Indexing a reference dataset returns a h5py.Reference instance """
        dset = self.f.create_dataset(make_name(), (1,), dtype=h5py.ref_dtype)
        dset[0] = self.f.ref
        self.assertEqual(type(dset[0]), h5py.Reference)

    def test_regref(self):
        """ Indexing a region reference dataset returns a h5py.RegionReference
        """
        dset1 = self.f.create_dataset(make_name("x"), (10,10), "f4")
        regref = dset1.regionref[...]
        dset2 = self.f.create_dataset(make_name("y"), (1,), dtype=h5py.regionref_dtype)
        dset2[0] = regref
        self.assertEqual(type(dset2[0]), h5py.RegionReference)

    def test_reference_field(self):
        """ Compound types of which a reference is an element work right """
        dt = np.dtype([('a', 'i'),('b', h5py.ref_dtype)])

        dset = self.f.create_dataset(make_name(), (1,), dtype=dt)
        dset[0] = (42, self.f['/'].ref)

        out = dset[0]
        self.assertEqual(type(out[1]), h5py.Reference)  # isinstance does NOT work

    def test_scalar(self):
        """ Indexing returns a real Python object on scalar datasets """
        dset = self.f.create_dataset(make_name(), (), dtype=h5py.ref_dtype)
        dset[()] = self.f.ref
        self.assertEqual(type(dset[()]), h5py.Reference)

    def test_bytestr(self):
        """ Indexing a byte string dataset returns a real python byte string
        """
        dset = self.f.create_dataset(make_name(), (1,), dtype=h5py.string_dtype(encoding='ascii'))
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

    def test_write(self):
        """Assigning to a 1D slice of a 2D dataset
        """
        dset = self.f.create_dataset(make_name(), (10, 2), "f4")

        x = np.zeros((10, 1))
        dset[:, 0] = x[:, 0]
        with self.assertRaises(TypeError):
            dset[:, 1] = x

class TestArraySlicing(BaseSlicing):

    """
        Feature: Array types are handled appropriately
    """

    def test_read(self):
        """ Read arrays tack array dimensions onto end of shape tuple """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset(make_name(), (10,), dtype=dt)
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

        dset = self.f.create_dataset(make_name(), (10,), dtype=dt)

        with self.assertRaises(TypeError):
            dset[...] = 42

    def test_write_element(self):
        """ Write a single element to the array

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset(make_name(), (10,), dtype=dt)

        data = np.array([1,2,3.0])
        dset[4] = data

        out = dset[4]
        self.assertTrue(np.all(out == data))

    def test_write_slices(self):
        """ Write slices to array type """
        dt = np.dtype('(3,)i')

        data1 = np.ones((2,), dtype=dt)
        data2 = np.ones((4,5), dtype=dt)

        dset = self.f.create_dataset(make_name(), (10,9,11), dtype=dt)

        dset[0,0,2:4] = data1
        self.assertArrayEqual(dset[0,0,2:4], data1)

        dset[3, 1:5, 6:11] = data2
        self.assertArrayEqual(dset[3, 1:5, 6:11], data2)


    def test_roundtrip(self):
        """ Read the contents of an array and write them back

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset(make_name(), (10,), dtype=dt)

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
            dset = self.f.create_dataset(make_name(f"x{i}"), shape, dtype=int, maxshape=(None,)*len(shape))
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
            dset = self.f.create_dataset(make_name(f"x{i}"), shape, dtype=int, maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (1,)+shape[1:])

    def test_slice_of_length_zero(self):
        """ Get a slice of length zero from a non-empty dataset """
        for i, shape in enumerate([(3,), (2, 2,), (2,  1, 5)]):
            dset = self.f.create_dataset(make_name(f"x{i}"), data=np.zeros(shape, int), maxshape=(None,)*len(shape))
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
        """ Test read with field selections """
        self.assertArrayEqual(self.dset['a'], self.data['a'])

    def test_unicode_names(self):
        """ Unicode field names for for read and write """
        self.assertArrayEqual(self.dset['a'], self.data['a'])

        data = self.data.copy()
        dset = self.f.create_dataset(make_name(), data=data)
        dset['a'] = 42
        data['a'] = 42
        self.assertArrayEqual(dset['a'], data['a'])

    def test_write(self):
        """ Test write with field selections """
        data = self.data.copy()
        dset = self.f.create_dataset(make_name(), data=data)

        data['a'] *= 2
        dset['a'] = data
        self.assertTrue(np.all(dset[...] == data))
        data['b'] *= 4
        dset['b'] = data
        self.assertTrue(np.all(dset[...] == data))
        data['a'] *= 3
        data['c'] *= 3
        dset['a','c'] = data
        self.assertTrue(np.all(dset[...] == data))

    def test_write_noncompound(self):
        """ Test write with non-compound source (single-field) """
        data = self.data.copy()
        dset = self.f.create_dataset(make_name(), data=data)

        data['b'] = 1.0
        dset['b'] = 1.0
        self.assertTrue(np.all(dset[...] == data))


class TestMultiBlockSlice(BaseSlicing):

    def setUp(self):
        super().setUp()
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def test_default(self):
        # Default selects entire dataset as one block
        mbslice = MultiBlockSlice()

        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_default_explicit(self):
        mbslice = MultiBlockSlice(start=0, count=10, stride=1, block=1)

        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_start(self):
        mbslice = MultiBlockSlice(start=4)

        self.assertEqual(mbslice.indices(10), (4, 1, 6, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([4, 5, 6, 7, 8, 9]))

    def test_count(self):
        mbslice = MultiBlockSlice(count=7)

        self.assertEqual(mbslice.indices(10), (0, 1, 7, 1))
        np.testing.assert_array_equal(
            self.dset[mbslice], np.array([0, 1, 2, 3, 4, 5, 6])
        )

    def test_count_more_than_length_error(self):
        mbslice = MultiBlockSlice(count=11)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_stride(self):
        mbslice = MultiBlockSlice(stride=2)

        self.assertEqual(mbslice.indices(10), (0, 2, 5, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 2, 4, 6, 8]))

    def test_stride_zero_error(self):
        with self.assertRaises(ValueError):
            # This would cause a ZeroDivisionError if not caught
            MultiBlockSlice(stride=0, block=0).indices(10)

    def test_stride_block_equal(self):
        mbslice = MultiBlockSlice(stride=2, block=2)

        self.assertEqual(mbslice.indices(10), (0, 2, 5, 2))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_block_more_than_stride_error(self):
        with self.assertRaises(ValueError):
            MultiBlockSlice(block=3)

        with self.assertRaises(ValueError):
            MultiBlockSlice(stride=2, block=3)

    def test_stride_more_than_block(self):
        mbslice = MultiBlockSlice(stride=3, block=2)

        self.assertEqual(mbslice.indices(10), (0, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 1, 3, 4, 6, 7]))

    def test_block_overruns_extent_error(self):
        # If fully described then must fit within extent
        mbslice = MultiBlockSlice(start=2, count=2, stride=5, block=4)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_fully_described(self):
        mbslice = MultiBlockSlice(start=1, count=2, stride=5, block=4)

        self.assertEqual(mbslice.indices(10), (1, 5, 2, 4))
        np.testing.assert_array_equal(
            self.dset[mbslice], np.array([1, 2, 3, 4, 6, 7, 8, 9])
        )

    def test_count_calculated(self):
        # If not given, count should be calculated to select as many full blocks as possible
        mbslice = MultiBlockSlice(start=1, stride=3, block=2)

        self.assertEqual(mbslice.indices(10), (1, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([1, 2, 4, 5, 7, 8]))

    def test_zero_count_calculated_error(self):
        # In this case, there is no possible count to select even one block, so error
        mbslice = MultiBlockSlice(start=8, stride=4, block=3)

        with self.assertRaises(ValueError):
            mbslice.indices(10)


# Broadcasting a small value over a big selection writes through a buffer of
# whole chunks (issue #1067). A tiny buffer limit makes these small datasets
# use several blocks, including partial ones at the edges.
BROADCAST_CASES = [
    # shape, chunks, selection, value shape (None: Python scalar)
    ((10, 11), (3, 4), np.s_[...], None),
    ((10, 11), (3, 4), np.s_[...], (11,)),
    ((10, 11), (3, 4), np.s_[...], (10, 1)),
    ((10, 11), (3, 4), np.s_[...], (1, 1, 11)),
    ((10, 11), (3, 4), np.s_[1:9, 2:10], None),
    ((10, 11), (3, 4), np.s_[1:9, 2:10], (8,)),
    ((10, 11), (3, 4), np.s_[::3, 1::2], None),
    ((10, 11), (2, 2), np.s_[1::5, ::3], (4,)),
    ((10, 11), (3, 4), np.s_[4, 1:], None),
    ((10, 11), (3, 4), np.s_[2:7, 5], None),
    ((10, 11), (10, 11), np.s_[...], None),
    ((10, 11), (1, 11), np.s_[...], (11,)),
    ((9, 8, 7), (2, 3, 4), np.s_[...], None),
    ((9, 8, 7), (2, 3, 4), np.s_[1:, :, 2:7], (5,)),
    ((9, 8, 7), (2, 3, 4), np.s_[1:, :, 2:7], (8, 1)),
    ((9, 8, 7), (2, 3, 4), np.s_[1:, 3, 2:7], (1, 5)),
    ((9, 8, 7), (4, 8, 1), np.s_[::2, 1:7, ::3], None),
    ((9, 8, 7), None, np.s_[...], None),
    ((9, 8, 7), None, np.s_[1:8, 2:, 1:6], (5,)),
    ((40,), (7,), np.s_[3:38], None),
    ((40,), None, np.s_[::3], None),
]


@pytest.mark.thread_unsafe(reason="monkey-patch")
@pytest.mark.parametrize("max_nbytes", [1, 100, 2**24])
@pytest.mark.parametrize("shape, chunks, selection, value_shape", BROADCAST_CASES)
def test_broadcast_write(writable_file, monkeypatch, max_nbytes,
                         shape, chunks, selection, value_shape):
    monkeypatch.setattr(dataset_mod, "_BROADCAST_BUFFER_NBYTES", max_nbytes)
    initial = np.arange(np.prod(shape), dtype='f8').reshape(shape)
    dset = writable_file.create_dataset(make_name(), data=initial, chunks=chunks)

    if value_shape is None:
        value = -1.5
    else:
        value = -np.arange(1, np.prod(value_shape) + 1).reshape(value_shape)
    dset[selection] = value

    expected = initial.copy()
    expected[selection] = value
    np.testing.assert_array_equal(dset[()], expected)


@pytest.mark.thread_unsafe(reason="monkey-patch")
def test_broadcast_write_converts_like_full_array(writable_file, monkeypatch):
    # Type conversion is left to HDF5, as when writing a full array
    monkeypatch.setattr(dataset_mod, "_BROADCAST_BUFFER_NBYTES", 1)
    value = np.array([0.4, 2.6, -3.7, 1e3], dtype='f4')
    dset = writable_file.create_dataset(make_name(), (12, 4), dtype='i2', chunks=(5, 3))
    ref = writable_file.create_dataset(make_name("ref"), (12, 4), dtype='i2', chunks=(5, 3))

    dset[1:] = value
    ref[1:] = np.broadcast_to(value, (11, 4)).copy()
    np.testing.assert_array_equal(dset[()], ref[()])


@pytest.mark.thread_unsafe(reason="monkey-patch")
@pytest.mark.parametrize("dtype, value", [
    (np.dtype('S5'), b'abc'),
    (np.dtype([('a', 'i4'), ('b', 'f8')]), np.array((3, 2.5), dtype=[('a', 'i4'), ('b', 'f8')])),
    (h5py.string_dtype(), 'variable'),
    (h5py.vlen_dtype('i4'), np.array([1, 2, 3], dtype='i4')),
])
def test_broadcast_write_other_dtypes(writable_file, monkeypatch, dtype, value):
    monkeypatch.setattr(dataset_mod, "_BROADCAST_BUFFER_NBYTES", 1)
    dset = writable_file.create_dataset(make_name(), (7, 6), dtype=dtype, chunks=(2, 4))
    dset[1:, 2:] = value

    if dtype.kind == 'O':
        # Variable-length data is written by the general broadcasting code
        cells = list(dset.asstr()[1:, 2:].ravel()) if isinstance(value, str) \
            else [x.tolist() for x in dset[1:, 2:].ravel()]
        expected = value if isinstance(value, str) else value.tolist()
        assert cells == [expected] * 24
    else:
        expected = np.zeros((7, 6), dtype=dtype)
        expected[1:, 2:] = value
        np.testing.assert_array_equal(dset[()], expected)


@pytest.mark.thread_unsafe(reason="monkey-patch")
def test_broadcast_write_compound_field(writable_file, monkeypatch):
    monkeypatch.setattr(dataset_mod, "_BROADCAST_BUFFER_NBYTES", 1)
    dt = np.dtype([('a', 'i4'), ('b', 'f8')])
    dset = writable_file.create_dataset(make_name(), (9, 5), dtype=dt, chunks=(2, 2))
    dset['b'] = 1.5
    dset[2:, 'a'] = np.arange(5)

    out = dset[()]
    np.testing.assert_array_equal(out['b'], np.full((9, 5), 1.5))
    expected_a = np.zeros((9, 5), dtype='i4')
    expected_a[2:] = np.arange(5)
    np.testing.assert_array_equal(out['a'], expected_a)


def _tile_blocks(shape, selection, source_shape, max_nbytes, chunks, itemsize=8):
    sel = h5py._hl.selections.select(shape, selection)
    plan = sel.broadcast_tiles(source_shape, itemsize, max_nbytes, chunks)
    if plan is None:
        return None, None
    tile, blocks = plan
    out = []
    for count, fspace in blocks:
        assert all(c <= t for c, t in zip(count, tile))
        start, _ = fspace.get_select_bounds()
        out.append((start, count, fspace.get_select_npoints()))
    assert sum(n for _, _, n in out) == sel.nselect
    return tile, out


def test_broadcast_tiles_follow_chunks():
    # Scalar over (100, 100) with (10, 10) chunks: the tile is one chunk
    # high and grows along the last axis within the limit.
    tile, blocks = _tile_blocks((100, 100), np.s_[...], (), 1600, (10, 10))
    assert tile == (10, 20)
    assert len(blocks) == 50
    assert all(s[0] % 10 == 0 and s[1] % 10 == 0 for s, _, _ in blocks)

    # With room for everything, the whole selection is one write
    tile, blocks = _tile_blocks((100, 100), np.s_[...], (100,), 2**20, (10, 10))
    assert tile == (100, 100)
    assert len(blocks) == 1

    # Unaligned selection: blocks after the first start on chunk boundaries
    tile, blocks = _tile_blocks((100, 100), np.s_[3:97, 5:95], (), 800, (10, 10))
    assert tile == (10, 10)
    starts0 = sorted({s[0] for s, _, _ in blocks})
    starts1 = sorted({s[1] for s, _, _ in blocks})
    assert starts0 == [3] + list(range(10, 97, 10))
    assert starts1 == [5] + list(range(10, 95, 10))

    # Contiguous dataset: whole rows, as many as fit
    tile, blocks = _tile_blocks((100, 100), np.s_[...], (), 8000, None)
    assert tile == (10, 100)
    assert len(blocks) == 10


def test_broadcast_tiles_not_used():
    # Nothing to repeat
    assert _tile_blocks((10, 10), np.s_[...], (10, 10), 1, (2, 2)) == (None, None)
    # MultiBlockSlice selections
    mbs = MultiBlockSlice(start=1, stride=3, count=3, block=2)
    assert _tile_blocks((10, 10), np.s_[mbs, :], (), 1, (2, 2)) == (None, None)
    # No buffer bigger than the source fits
    assert _tile_blocks((10, 10), np.s_[...], (10,), 8, (1, 1)) == (None, None)
