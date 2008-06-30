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
import sys
import test_h5a, test_h5d, test_h5f, test_h5g, test_h5i, test_h5p

from h5py import h5a, h5f, h5g, h5d, h5s, h5i, h5z, h5p

TEST_CASES = (test_h5a.TestH5A, test_h5d.TestH5D, test_h5f.TestH5F, 
              test_h5g.TestH5G, test_h5i.TestH5I, test_h5p.TestH5P)#, test_h5.TestH5,
              #test_h5s.TestH5S, test_highlevel.TestHighlevel)

def buildsuite(cases):

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return suite

def runtests():
    suite = buildsuite(TEST_CASES)
    retval = unittest.TextTestRunner(verbosity=3).run(suite)
    return retval.wasSuccessful()

def autotest():
    if not runtests():
        sys.exit(1)



