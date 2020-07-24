# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Common high-level operations test

    Tests features common to all high-level objects, like the .name property.
"""

from h5py import File
from h5py._hl.base import is_hdf5, Empty
from .common import ut, TestCase, UNICODE_FILENAMES

import numpy as np
import os
import tempfile

class BaseTest(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestName(BaseTest):

    """
        Feature: .name attribute returns the object name
    """

    def test_anonymous(self):
        """ Anonymous objects have name None """
        grp = self.f.create_group(None)
        self.assertIs(grp.name, None)

class TestParent(BaseTest):

    """
        test the parent group of the high-level interface objects
    """

    def test_object_parent(self):
        # Anonymous objects
        try:
            grp = self.f.create_group(None)
            bar_parent = grp.parent
        except ValueError:
            pass

        # Named objects
        grp = self.f.create_group("bar")
        sub_grp = grp.create_group("foo")
        parent = sub_grp.parent
        self.assertEqual(r'<HDF5 group "/bar" (1 members)>', repr(parent))

class TestMapping(BaseTest):

    """
        Test if the registration of Group, AttributeManager as a
        Mapping behaves as expected
    """

    def setUp(self):
        data = ('a', 'b')
        self.f = File('foo.hdf5', 'w')
        self.grp = self.f.create_group('bar')
        self.attr = self.f.attrs.create('x', data)

    def TearDown(self):
        if self.f:
            self.close()

    def test_keys(self):
        key_1 = self.f.keys()
        self.assertEqual(r"<KeysViewHDF5 ['bar']>", repr(key_1))
        key_2 = self.grp.keys()
        self.assertEqual(r"<KeysViewHDF5 []>", repr(key_2))

    def test_values(self):
        value_1 = self.f.values()
        self.assertEqual(r'ValuesViewHDF5(<HDF5 file "foo.hdf5" (mode r+)>)', repr(value_1))
        value_2 = self.grp.values()
        self.assertEqual(r'ValuesViewHDF5(<HDF5 group "/bar" (0 members)>)', repr(value_2))

    def test_items(self):
        item_1 = self.f.items()
        self.assertEqual(r'ItemsViewHDF5(<HDF5 file "foo.hdf5" (mode r+)>)', repr(item_1))
        item_2 = self.grp.items()
        self.assertEqual(r'ItemsViewHDF5(<HDF5 group "/bar" (0 members)>)', repr(item_2))

class TestFileType(BaseTest):

    """
        Test if a file is a HDF5 type
    """

    def test_is_hdf5(self):
        filename = File("data.hdf5", "w").filename
        fname = os.path.basename(filename)
        fid = is_hdf5(fname)
        self.assertTrue(fid)
        # non-existing HDF5 file
        filename = tempfile.mktemp()
        fname = os.path.basename(filename)
        fid = is_hdf5(fname)
        self.assertFalse(fid)

class TestRepr(BaseTest):

    """
        repr() works correctly with Unicode names
    """

    USTRING = chr(0xfc) + chr(0xdf)

    def _check_type(self, obj):
        self.assertIsInstance(repr(obj), str)

    def test_group(self):
        """ Group repr() with unicode """
        grp = self.f.create_group(self.USTRING)
        self._check_type(grp)

    def test_dataset(self):
        """ Dataset repr() with unicode """
        dset = self.f.create_dataset(self.USTRING, (1,))
        self._check_type(dset)

    def test_namedtype(self):
        """ Named type repr() with unicode """
        self.f['type'] = np.dtype('f')
        typ = self.f['type']
        self._check_type(typ)

    def test_empty(self):
        data = Empty(dtype='f')
        self.assertNotEqual(Empty(dtype='i'), data)
        self._check_type(data)

    @ut.skipIf(not UNICODE_FILENAMES, "Filesystem unicode support required")
    def test_file(self):
        """ File object repr() with unicode """
        fname = tempfile.mktemp(self.USTRING+'.hdf5')
        try:
            with File(fname,'w') as f:
                self._check_type(f)
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass
