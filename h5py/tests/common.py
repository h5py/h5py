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

def skip(func):
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

EPSILON = 1e-5
import numpy as np
from nose.tools import assert_equal

INTS = ('i', 'i1', '<i2', '>i2', '<i4', '>i4')
FLOATS = ('f', '<f4', '>f4', '<f8', '>f8')
COMPLEX = ('<c8', '>c8', '<c16', '>c16')
STRINGS = ('|S1', '|S2', 'S17', '|S100')
VOIDS = ('|V4', '|V8')

def assert_arr_equal(dset, arr, message=None, precision=None):
    """ Make sure dset and arr have the same shape, dtype and contents, to
        within the given precision.

        Note that dset may be a NumPy array or an HDF5 dataset.
    """
    if precision is None:
        precision = EPSILON
    if message is None:
        message = ''

    if np.isscalar(dset) or np.isscalar(arr):
        assert np.isscalar(dset) and np.isscalar(arr), "%r %r" % (dset, arr)
        assert dset - arr < precision, message
        return

    assert_equal(dset.shape, arr.shape, message)
    assert_equal(dset.dtype, arr.dtype, message)
    assert np.all(np.abs(dset[...] - arr[...]) < precision), "%s %s" % (dset[...], arr[...]) if not message else message

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



    

