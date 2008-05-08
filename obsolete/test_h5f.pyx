##### Preamble block ##########################################################
# 
# This file is part of the "h5py" HDF5 Interface for Python project.
# 
# Copyright 2008 Andrew Collette
# http://software.alfven.org
# License: BSD  (See file "LICENSE" for complete license, or the URL above)
# 
##### End preamble block ######################################################

from defs_h5f cimport H5Fopen, H5Fclose,\
                      H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
from defs_h5p cimport H5P_DEFAULT
from defs_h5i cimport H5Iget_type, H5I_FILE
                 
import unittest
import os
import tempfile

import h5f
from errors import FileError

"""
    Tests functions defined in h5f.  Requires HDF5 file; default name
    is "test_simple.hdf5".

"""
TEST_FILE = "test_simple.hdf5"

class TestH5F(unittest.TestCase):

    def testopen(self):
        os.chmod(TEST_FILE, 0600)

        fid = h5f.open(TEST_FILE, flags=h5f.ACC_RDWR)
        self.assert_(H5Iget_type(fid) == H5I_FILE)
        H5Fclose(fid)

        fid = h5f.open(TEST_FILE, flags=h5f.ACC_RDONLY)
        self.assert_(H5Iget_type(fid) == H5I_FILE)
        H5Fclose(fid)     

        os.chmod(TEST_FILE, 0400)
        
        fid = h5f.open(TEST_FILE, flags=H5F_ACC_RDONLY)
        self.assert_(H5Iget_type(fid) == H5I_FILE)
        H5Fclose(fid)     

        self.assertRaises(FileError, h5f.open, TEST_FILE, flags=h5f.ACC_RDWR)
        
    def testclose(self):
        os.chmod(TEST_FILE, 0600)
        fid = H5Fopen(TEST_FILE, H5F_ACC_RDWR, H5P_DEFAULT)
        self.assert_(H5Iget_type(fid) == H5I_FILE)
        h5f.close(fid)

    def testcreate(self):

        fd, name = tempfile.mkstemp('.hdf5')
        os.close(fd)

        fid = h5f.create(name, flags=h5f.ACC_TRUNC)
        self.assert_(H5Iget_type(fid) == H5I_FILE)
        H5Fclose(fid)   

        self.assertRaises(FileError, h5f.create, name, flags=h5f.ACC_EXCL)
        
        os.unlink(name)








