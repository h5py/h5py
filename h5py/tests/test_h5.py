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

from h5py import h5

class TestH5(unittest.TestCase):

    def test_version(self):
        # For 1.6 API
        tpl = h5.get_libversion()

        self.assertEqual(tpl, h5.HDF5_VERS_TPL)
        self.assertEqual("%d.%d.%d" % tpl, h5.HDF5_VERS)
        h5.API_VERS
        h5.API_VERS_TPL
