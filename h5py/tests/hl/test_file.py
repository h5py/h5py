# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.File object.
"""

from __future__ import absolute_import

import h5py
from h5py._hl.files import _drivers

from ..common import ut, TestCase


def nfiles():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)

def ngroups():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_GROUP)

        
class TestDealloc(TestCase):

    """ 
        Behavior on object deallocation.  Note most of this behavior is
        delegated to FileID.
    """
    
    def test_autoclose(self):
        """ File objects close automatically when out of scope, but
        other objects remain open. """
        
        start_nfiles = nfiles()
        start_ngroups = ngroups()
        
        fname = self.mktemp()
        f = h5py.File(fname, 'w')
        g = f['/']
        
        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups+1)
        
        del f
        
        self.assertTrue(g)
        self.assertEqual(nfiles(), start_nfiles)
        self.assertEqual(ngroups(), start_ngroups+1)
        
        f = g.file
        
        self.assertTrue(f)
        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups+1)
        
        del g
        
        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups)
        
        del f
        
        self.assertEqual(nfiles(), start_nfiles)
        self.assertEqual(ngroups(), start_ngroups)


class TestDriverRegsitration(TestCase):
    def test_register_driver(self):
        called_with = [None]

        def set_fapl(plist, *args, **kwargs):
            called_with[0] = args, kwargs
            return _drivers['sec2'](plist)

        h5py.register_driver('new-driver', set_fapl)
        self.assertIn('new-driver', h5py.registered_drivers())

        fname = self.mktemp()
        h5py.File(fname, driver='new-driver', driver_arg_0=0, driver_arg_1=1)

        self.assertEqual(
            called_with,
            [((), {'driver_arg_0': 0, 'driver_arg_1': 1})],
        )

    def test_unregister_driver(self):
        h5py.register_driver('new-driver', lambda plist: None)
        self.assertIn('new-driver', h5py.registered_drivers())

        h5py.unregister_driver('new-driver')
        self.assertNotIn('new-driver', h5py.registered_drivers())

        with self.assertRaises(ValueError) as e:
            fname = self.mktemp()
            h5py.File(fname, driver='new-driver')

        self.assertEqual(str(e.exception), 'Unknown driver type "new-driver"')
