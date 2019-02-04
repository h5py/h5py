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
import platform


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
'''.format((test.__qualname__.split('.')[0])
           if hasattr(test, '__qualname__') else 'Test',
           source, check=True))

            fname = f.name
            try:
                # support Windows
                f.close()

                env = dict(os.environ)
                # as per https://support.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf,
                # in case your HDF5 setup has it different
                # (e.g. https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=826522)
                env['HDF5_PLUGIN_PATH'] = os.path.expandvars('%ALLUSERSPROFILE%/hdf5/lib/plugin') \
                    if platform.system() == 'Windows' else '/usr/local/hdf5/lib/plugin'

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
        self.assertTrue(h5pl.get(0).endswith(b'hdf5/lib/plugin\x00'))

    @sandboxed
    def test_append(self):
        h5pl.append(b'/opt/hdf5/vendor-plugin')
        self.assertEqual(h5pl.size(), 2)
        self.assertTrue(h5pl.get(0).endswith(b'hdf5/lib/plugin\x00'))
        self.assertEqual(h5pl.get(1), b'/opt/hdf5/vendor-plugin\x00')

    @sandboxed
    def test_prepend(self):
        h5pl.prepend(b'/opt/hdf5/vendor-plugin')
        self.assertEqual(h5pl.size(), 2)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugin\x00')
        self.assertTrue(h5pl.get(1).endswith(b'hdf5/lib/plugin\x00'))

    @sandboxed
    def test_insert(self):
        h5pl.insert(b'/opt/hdf5/vendor-plugin', 0)
        self.assertEqual(h5pl.size(), 2)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugin\x00')
        self.assertTrue(h5pl.get(1).endswith(b'hdf5/lib/plugin\x00'))

    @sandboxed
    def test_replace(self):
        h5pl.replace(b'/opt/hdf5/vendor-plugin', 0)
        self.assertEqual(h5pl.size(), 1)
        self.assertEqual(h5pl.get(0), b'/opt/hdf5/vendor-plugin\x00')

    @sandboxed
    def test_remove(self):
        h5pl.remove(0)
        self.assertEqual(h5pl.size(), 0)
