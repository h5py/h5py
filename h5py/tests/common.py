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

import os
import unittest
import tempfile
import os.path as op
import shutil
from h5py import h5f, h5p, h5
import h5py

DATADIR = op.join(op.dirname(h5py.__file__), 'tests/data')

def getfullpath(name):
    return op.abspath(op.join(DATADIR, name))

def api_18(func):
    """Decorator to enable 1.8.X-only API functions"""
    if h5.get_config().API_18:
        return func
    return None

def api_16(func):
    """Decorator to run test under HDF5 1.6 only"""
    if not h5.get_config().API_18:
        return func
    return None

test_coverage = set()

def covers(*args):
    global test_coverage
    
    def wrap(meth):
        test_coverage.update(args)
        return meth

    return wrap

def makehdf():
    fname = tempfile.mktemp('.hdf5')
    f = h5py.File(fname, 'w')
    return f

def delhdf(f):
    fname = f.name
    f.close()
    os.unlink(fname)

class HDF5TestCase(unittest.TestCase):

    """
        Base test for unit test classes.
    """

    h5py_verbosity = 0

    def output(self, ipt):
        """Print to stdout, only if verbosity levels so requires"""
        if self.h5py_verbosity >= 3:
            print ipt

    def setup_fid(self, hdfname):
        """Open a copy of an HDF5 file and set its identifier as self.fid"""
        hdfname = getfullpath(hdfname)
        newname = tempfile.mktemp('.hdf5')
        shutil.copy(hdfname, newname)

        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        self.fid = h5f.open(newname, h5f.ACC_RDWR, fapl=plist)
        self.fname = newname
        self.src_fname = hdfname

    def teardown_fid(self):
        """Close the HDF5 file copy and delete it"""
        self.fid.close()
        os.unlink(self.fname)



    

