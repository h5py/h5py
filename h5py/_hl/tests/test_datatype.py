
"""
    File-resident datatype tests.

    Tests "committed" file-resident datatype objects.
"""

import numpy as np

from .common import ut, TestCase

from h5py import File
from h5py._hl.datatype import Datatype

class BaseType(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCreation(BaseType):

    """
        Feature: repr() works sensibly on datatype objects
    """

    def test_repr(self):
        """ repr() on datatype objects """
        self.f['foo'] = np.dtype('S10')
        dt = self.f['foo']
        self.assertIsInstance(repr(dt), basestring)
        self.f.close()
        self.assertIsInstance(repr(dt), basestring)


    def test_appropriate_low_level_id(self):
        " Binding a group to a non-TypeID identifier fails with ValueError "
        with self.assertRaises(ValueError):
            Datatype(self.f['/'].id)
