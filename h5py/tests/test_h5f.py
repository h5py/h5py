#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

import unittest
import tempfile
import os

import h5py
from h5py import h5f, h5i, h5p
from common import getcopy, deletecopy, errstr
from h5py.h5e import H5Error

HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/attributes.hdf5')

class TestH5F(unittest.TestCase):

    def setUp(self):
        self.fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)

    def tearDown(self):
        h5f.close(self.fid)

    def test_open_close(self):
        fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)
        self.assertEqual(h5i.get_type(fid), h5i.FILE)
        h5f.close(fid)
        self.assertEqual(h5i.get_type(fid), h5i.BADID)

        self.assertRaises(H5Error, h5f.open, 'SOME OTHER NAME')
        self.assertRaises(H5Error, h5f.close, -1)

    def test_create(self):
        name = tempfile.mktemp('.hdf5')
        fid = h5f.create(name)
        self.assertEqual(h5i.get_type(fid), h5i.FILE)
        h5f.close(fid)
        self.assertRaises(H5Error, h5f.create, name, h5f.ACC_EXCL)
        os.unlink(name)

    def test_flush(self):
        fid, fname = getcopy(HDFNAME)
        h5f.flush(fid)
        self.assertRaises(H5Error, h5f.flush, -1)
        deletecopy(fid, fname)

    def test_is_hdf5(self):
        fd, name = tempfile.mkstemp('.hdf5')
        os.close(fd)
        try:
            self.assert_(not h5f.is_hdf5(name))
        finally:
            os.unlink(name)

        self.assert_(h5f.is_hdf5(HDFNAME))

    def test_get_filesize(self):

        self.assertEqual(h5f.get_filesize(self.fid), os.stat(HDFNAME).st_size)
        self.assertRaises(H5Error, h5f.get_filesize, -1)

    def test_get_create_plist(self):
        cplist = h5f.get_create_plist(self.fid)
        self.assert_(h5p.equal(h5p.get_class(cplist), h5p.FILE_CREATE))
        h5p.close(cplist)
        self.assertRaises(H5Error, h5f.get_create_plist, -1)

    def test_get_access_plist(self):
        aplist = h5f.get_access_plist(self.fid)
        self.assert_(h5p.equal(h5p.get_class(aplist), h5p.FILE_ACCESS))
        h5p.close(aplist)
        self.assertRaises(H5Error, h5f.get_access_plist, -1)

    def test_get_freespace(self):
        self.assert_(h5f.get_freespace(self.fid) >= 0)
        self.assertRaises(H5Error, h5f.get_freespace, -1)

    def test_get_name(self):
        self.assertEqual(h5f.get_name(self.fid), HDFNAME)
        self.assertRaises(H5Error, h5f.get_name, -1)

    def test_get_obj_count(self):
        self.assert_(h5f.get_obj_count(self.fid, h5f.OBJ_ALL) >= 0)
        self.assertRaises(H5Error, h5f.get_obj_count, -1, h5f.OBJ_ALL)
    
    def test_get_obj_ids(self):
        idlist = h5f.get_obj_ids(self.fid, h5f.OBJ_ALL)
        self.assert_(isinstance(idlist, list))
        self.assertRaises(H5Error, h5f.get_obj_ids, -1, h5f.OBJ_ALL)




    
        
