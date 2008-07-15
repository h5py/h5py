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
from os.path import join, dirname
import shutil
from h5py import h5f, h5p
import h5py

DATADIR = join(dirname(h5py.__file__), 'tests/data')

class TestBase(unittest.TestCase):

    """
        Base test for unit test classes which deal with a particular
        HDF5 file.  These should declare a class-level attribute HDFNAME
        representing the appropriate file.  This should be a basename; the
        TestBase class will figure out the correct directory.
    """

    def __init__(self, *args, **kwds):
        unittest.TestCase.__init__(self, *args, **kwds)
        self.HDFNAME = join(DATADIR, self.HDFNAME) # resolve absolute location

    def setUp(self):
        newname = tempfile.mktemp('.hdf5')
        shutil.copy(self.HDFNAME, newname)

        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        self.fid = h5f.open(newname, h5f.ACC_RDWR, accesslist=plist)
        self.fname = newname

    def tearDown(self):
        self.fid.close()
        os.unlink(self.fname)

