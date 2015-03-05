# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from __future__ import absolute_import

import sys
import os
import shutil
import tempfile

from six import unichr

import numpy as np
import h5py

if sys.version_info >= (2, 7) or sys.version_info >= (3, 2):
    import unittest as ut
else:
    try:
        import unittest2 as ut
    except ImportError:
        raise ImportError(
            'unittest2 is required to run the test suite with python-%d.%d'
            % (sys.version_info[:2])
            )


# Check if non-ascii filenames are supported
# Evidently this is the most reliable way to check
# See also h5py issue #263 and ipython #466
# To test for this, run the testsuite with LC_ALL=C
try:
    testfile, fname = tempfile.mkstemp(unichr(0x03b7))
except UnicodeError:
    UNICODE_FILENAMES = False
else:
    UNICODE_FILENAMES = True
    os.close(testfile)
    os.unlink(fname)
    del fname
    del testfile


class TestCase(ut.TestCase):

    """
        Base class for unit tests.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix='h5py-test_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def mktemp(self, suffix='.hdf5', prefix='', dir=None):
        if dir is None:
            dir = self.tempdir
        return tempfile.mktemp(suffix, prefix, dir=self.tempdir)
        
    def setUp(self):
        self.f = h5py.File(self.mktemp(), 'w')
        
    def tearDown(self):
        try:
            if self.f:
                self.f.close()
        except:
            pass

    if not hasattr(ut.TestCase, 'assertSameElements'):
        # shim until this is ported into unittest2
        def assertSameElements(self, a, b):
            for x in a:
                match = False
                for y in b:
                    if x == y:
                        match = True
                if not match:
                    raise AssertionError("Item '%s' appears in a but not b" % x)

            for x in b:
                match = False
                for y in a:
                    if x == y:
                        match = True
                if not match:
                    raise AssertionError("Item '%s' appears in b but not a" % x)

    def assertArrayEqual(self, dset, arr, message=None, precision=None):
        """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
        if precision is None:
            precision = 1e-5
        if message is None:
            message = ''
        else:
            message = ' (%s)' % message

        if np.isscalar(dset) or np.isscalar(arr):
            self.assert_(
                np.isscalar(dset) and np.isscalar(arr),
                'Scalar/array mismatch ("%r" vs "%r")%s' % (dset, arr, message)
                )
            self.assert_(
                dset - arr < precision,
                "Scalars differ by more than %.3f%s" % (precision, message)
                )
            return

        self.assert_(
            dset.shape == arr.shape,
            "Shape mismatch (%s vs %s)%s" % (dset.shape, arr.shape, message)
            )
        self.assert_(
            dset.dtype == arr.dtype,
            "Dtype mismatch (%s vs %s)%s" % (dset.dtype, arr.dtype, message)
            )
            
        if arr.dtype.names is not None:
            for n in arr.dtype.names:
                message = '[FIELD %s] %s' % (n, message)
                self.assertArrayEqual(dset[n], arr[n], message=message, precision=precision)
        elif arr.dtype.kind in ('i', 'f'):
            self.assert_(
                np.all(np.abs(dset[...] - arr[...]) < precision),
                "Arrays differ by more than %.3f%s" % (precision, message)
                )
        else:
            self.assert_(
                np.all(dset[...] == arr[...]),
                "Arrays are not equal (dtype %s) %s" % (arr.dtype.str, message)
                )

    def assertNumpyBehavior(self, dset, arr, s):
        """ Apply slicing arguments "s" to both dset and arr.
        
        Succeeds if the results of the slicing are identical, or the
        exception raised is of the same type for both.
        
        "arr" must be a Numpy array; "dset" may be a NumPy array or dataset.
        """
        exc = None
        try:
            arr_result = arr[s]
        except Exception as e:
            exc = type(e)
            
        if exc is None:
            self.assertArrayEqual(dset[s], arr_result)
        else:
            with self.assertRaises(exc):
                dset[s]
