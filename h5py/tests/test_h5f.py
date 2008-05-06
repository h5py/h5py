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
from h5py import h5f, h5i
from h5py.errors import FileError
from common import getcopy, deletecopy, errstr

HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/attributes.hdf5')

class TestH5F(unittest.TestCase):

    def test_open_close(self):
        fid = h5f.open(HDFNAME)
        self.assertEqual(h5i.get_type(fid), h5i.TYPE_FILE)
        h5f.close(fid)
        self.assertEqual(h5i.get_type(fid), h5i.TYPE_BADID)

        self.assertRaises(FileError, h5f.open, 'SOME OTHER NAME')
        self.assertRaises(FileError, h5f.close, -1)

    def test_create(self):
        name = tempfile.mktemp('.hdf5')
        fid = h5f.create(name)
        self.assertEqual(h5i.get_type(fid), h5i.TYPE_FILE)
        h5f.close(fid)
        self.assertRaises(FileError, h5f.create, name, h5f.ACC_EXCL)
        os.unlink(name)

    def test_flush(self):
        fid = h5f.open(HDFNAME, h5f.ACC_RDWR)
        h5f.flush(fid)
        self.assertRaises(FileError, h5f.flush, -1)
        h5f.close(fid)

    def test_is_hdf5(self):
        fd, name = tempfile.mkstemp('.hdf5')
        os.close(fd)
        try:
            self.assert_(not h5f.is_hdf5(name))
        finally:
            os.unlink(name)

        self.assert_(h5f.is_hdf5(HDFNAME))








    
        
