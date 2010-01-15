from __future__ import with_statement

"""
    Test highlevel Group object behavior
"""
import numpy as np

import h5py

from common import TestCasePlus, api_16, api_18, res
from common import dump_warnings

SHAPES = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]

class GroupBase(TestCasePlus):

    """
        Base class to handle Group setup/teardown, and some shared logic.
    """

    def setUp(self):
        self.f = h5py.File(res.get_name(), 'w')

    def tearDown(self):
        res.clear()

    def assert_equal_contents(self, a, b):
        """ Check if two iterables contain the same elements, regardless of
            order.
        """
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(a), len(b))

## -----

from h5py import Group, Dataset, Datatype, File
import numpy

class TestOther(GroupBase):

    def test_require(self):

        grp = self.f.require_group('foo')
        self.assert_(isinstance(grp, Group))
        self.assert_('foo' in self.f)

        grp2 = self.f.require_group('foo')
        self.assert_(grp == grp2)
        self.assert_(hash(grp) == hash(grp2))

        dset = self.f.require_dataset('bar', (10,10), '<i4')
        self.assert_(isinstance(dset, Dataset))
        self.assert_('bar' in self.f)

        dset2 = self.f.require_dataset('bar', (10,10), '<i4')
        self.assert_(dset == dset2)
        self.assert_(hash(dset) == hash(dset2))

        self.assertRaises(TypeError, self.f.require_group, 'bar')
        self.assertRaises(TypeError, self.f.require_dataset, 'foo', (10,10), '<i4')

        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,11), '<i4')
        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,10), '<c8')
        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,10), '<i1', exact=True)

        self.f.require_dataset('bar', (10,10), '<i1')

    @api_16
    def test_copy_16(self):

        self.f.create_group('foo')
        self.assertRaises(NotImplementedError, self.f.copy, 'foo', 'bar')

    @api_18
    def test_copy_18(self):

        self.f.create_group('foo')
        self.f.create_group('foo/bar')

        self.f.copy('foo', 'new')
        self.assert_('new' in self.f)
        self.assert_('new/bar' in self.f)

