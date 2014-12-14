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

from six import unichr, PY3

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

import shutil
import tempfile
import numpy as np
import os


class TestCase(ut.TestCase):

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
        self.assert_(
            np.all(np.abs(dset[...] - arr[...]) < precision),
            "Arrays differ by more than %.3f%s" % (precision, message)
            )

# Check if non-ascii filenames are supported
# Evidently this is the most reliable way to check
# See also h5py issue #263 and ipython #466
# To test for this, run the testsuite with LC_ALL=C
try:
    testfile, fname = tempfile.mkstemp(unichr(0x03b7))
except UnicodeError:
    unicode_filenames = False
else:
    unicode_filenames = True
    os.close(testfile)
    os.unlink(fname)
    del fname
    del testfile
