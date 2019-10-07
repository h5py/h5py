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
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile

import numpy as np


class TestName(TestCase):

    """
        Feature: .name attribute returns the object name
    """

    def test_anonymous(self):
        """ Anonymous objects have name None """
        grp = self.f.create_group(None)
        self.assertIs(grp.name, None)

class TestRepr(TestCase):

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

    @ut.skipIf(not UNICODE_FILENAMES, "Filesystem unicode support required")
    def test_file(self):
        """ File object repr() with unicode """
        with closed_tempfile(suffix=self.USTRING) as fname:
            with File(fname,'w') as f:
                self._check_type(f)
