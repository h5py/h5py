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

from ..common import ut, TestCase


def nfiles():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)

def ngroups():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_GROUP)

        
class TestDealloc(TestCase):

    """ 
        Behavior on object dellocation.  Note most of this behavior is
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
