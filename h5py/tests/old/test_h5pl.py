# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2019 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from __future__ import absolute_import

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import h5py
from h5py import h5pl

import inspect
import functools
import subprocess
import tempfile
import sys
import os


def sandboxed(test):

    @functools.wraps(test)
    def wrapper(*args):
        lines, start = inspect.getsourcelines(test)
        source = '\n'.join(lines[1:])  # remove this decorator

        with tempfile.NamedTemporaryFile('w', prefix='sandbox-', suffix='.py', delete=False) as f:
            f.write('''
try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import h5py
from h5py import h5pl

class {}(ut.TestCase):
{}
'''.format((test.__qualname__.split('.')[0]) if hasattr(test, '__qualname__')
           else test.im_class.__name__,
           source, check=True))

            fname = f.name
            try:
                # support Windows
                f.close()

                env = {**os.environ}
                if 'HDF5_PLUGIN_PATH' in env:
                    del env['HDF5_PLUGIN_PATH']

                subprocess.check_call((sys.executable, '-m', 'pytest', '-q', fname),
                                      env=env)
            finally:
                os.remove(fname)

    return wrapper


@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 1), 'HDF5 1.10.1+ required')
class TestSearchPaths(ut.TestCase):

    @sandboxed
    def test_default(self):
        self.assertEqual(h5pl.size(), 1)
        self.assertTrue(h5pl.get(0).endswith(b'hdf5/plugins\x00'))

    @sandboxed
    def test_append(self):
        h5pl.append(b'/opt/hdf5/vendor-plugins')
        self.assertEqual(h5pl.size(), 2)
        self.assertTrue(h5pl.get(0).endswith(b'hdf5/plugins\x00'))
        self.assertEqual(h5pl.get(1), b'/opt/hdf5/vendor-plugins\x00')

    @sandboxed
    def test_prepend(self):
        h5pl.prepend(b'/opt/hdf5/vendor-plugins')
        self.assertEqual(h5pl.size(), 2)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugins\x00')
        self.assertTrue(h5pl.get(1).endswith(b'hdf5/plugins\x00'))

    @sandboxed
    def test_insert(self):
        h5pl.insert(b'/opt/hdf5/vendor-plugins', 0)
        self.assertEqual(h5pl.size(), 2)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugins\x00')
        self.assertTrue(h5pl.get(1).endswith(b'hdf5/plugins\x00'))

    @sandboxed
    def test_replace(self):
        h5pl.replace(b'/opt/hdf5/vendor-plugins', 0)
        self.assertEqual(h5pl.size(), 1)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugins\x00')

    @sandboxed
    def test_remove(self):
        h5pl.remove(0)
        self.assertEqual(h5pl.size(), 0)
