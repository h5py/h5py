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

import numpy as np

DATADIR = op.join(op.dirname(h5py.__file__), 'tests/data')


import warnings
from contextlib import contextmanager

@contextmanager
def dump_warnings():
    filters = warnings.filters
    warnings.simplefilter("ignore")
    yield
    warnings.filters = filters
    
class ResourceManager(object):

    """
        Implements common operations, including generating filenames,
        files, and cleaning up identifiers.  This frees each module from
        having to manually unlink its files, and restores the library to
        a known state.
    """
    
    def __init__(self):
        self.fnames = set()

    def get_name(self, suffix=None):
        """ Return a temporary filename, which can be unlinked with clear() """
        if suffix is None:
            suffix = '.hdf5'
        fname = tempfile.mktemp(suffix)
        self.fnames.add(fname)
        return fname

    def clear(self):
        """ Wipe out all open identifiers, and unlink all generated files """
        id_list = h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_ALL^h5py.h5f.OBJ_DATATYPE)

        for id_ in id_list:
            while(id_ and h5py.h5i.get_ref(id_) > 0):
                h5py.h5i.dec_ref(id_)

        for fname in self.fnames:
            if op.exists(fname):
                os.unlink(fname)
            try:
                fname %= 0
            except TypeError:
                continue
            if op.exists(fname):
                os.unlink(fname)

        self.fnames.clear()

    def get_data_path(self, name):
        """ Return the full path to a data file (given its basename) """
        return op.abspath(op.join(DATADIR, name))

    def get_data_copy(self, name):
        iname = self.get_data_path(name)
        fname = self.get_name()
        shutil.copy(iname, fname)
        return fname

res = ResourceManager()

class TypeManager(object):

    ints =  [np.dtype(x) for x in ('i', 'i1', '<i2', '>i2', '<i4', '>i4')]
    uints = [np.dtype(x) for x in ('u1', '<u2', '>u2', '<u4', '>u4')]
    floats =  [np.dtype(x) for x in ('f', '<f4', '>f4', '<f8', '>f8')]
    complex = [np.dtype(x) for x in ('<c8', '>c8', '<c16', '>c16')]
    strings = [np.dtype(x) for x in ('|S1', '|S2', 'S17', '|S100')]
    voids =   [np.dtype(x) for x in ('|V1', '|V4', '|V8', '|V193')]

    compounds = [np.dtype(x) for x in \
                (   [('a', 'i'), ('b', 'f')],
                    [('a', '=c8'), ('b', [('a', 'i'), ('b', 'f')])] ) ] 

types = TypeManager()

FLOATS = ('f', '<f4', '>f4', '<f8', '>f8')
COMPLEX = ('<c8', '>c8', '<c16', '>c16')
STRINGS = ('|S1', '|S2', 'S17', '|S100')
VOIDS = ('|V4', '|V8')


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

skipped = []
def skip(func):
    skipped.append(func)
    return None


EPSILON = 1e-5
import numpy as np

INTS = ('i', 'i1', '<i2', '>i2', '<i4', '>i4')
UINTS = ('u1', '<u2', '>u2', '<u4', '>u4')
FLOATS = ('f', '<f4', '>f4', '<f8', '>f8')
COMPLEX = ('<c8', '>c8', '<c16', '>c16')
STRINGS = ('|S1', '|S2', 'S17', '|S100')
VOIDS = ('|V4', '|V8')

class TestCasePlus(unittest.TestCase):

    def setup_fid(self, name):
        self.fname = res.get_data_copy(name)
        
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        self.fid = h5f.open(self.fname, h5f.ACC_RDWR, fapl=plist)

    def teardown_fid(self):
        self.fid.close()
        os.unlink(self.fname)

    def assertRaisesMsg(self, msg, exc, clb, *args, **kwds):
        try:
            clb(*args, **kwds)
        except exc:
            return
        raise AssertionError("%s not raised: %s" % (exc, msg))

    def assertArrayEqual(self, dset, arr, message=None, precision=None):
        """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
        if precision is None:
            precision = EPSILON
        if message is None:
            message = ''
        else:
            message = ' (%s)' % message

        if np.isscalar(dset) or np.isscalar(arr):
            assert np.isscalar(dset) and np.isscalar(arr), 'Scalar/array mismatch ("%r" vs "%r")%s' % (dset, arr, message)
            assert dset - arr < precision, "Scalars differ by more than %.3f%s" % (precision, message)
            return

        assert dset.shape == arr.shape, "Shape mismatch (%s vs %s)%s" % (dset.shape, arr.shape, message)
        assert dset.dtype == arr.dtype, "Dtype mismatch (%s vs %s)%s" % (dset.dtype, arr.dtype, message)
        assert np.all(np.abs(dset[...] - arr[...]) < precision), "Arrays differ by more than %.3f%s" % (precision, message)








    

