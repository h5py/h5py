
"""
    Package for testing low-level interface to h5py
"""

import unittest

import test_h5, test_h5a, test_h5d, test_h5f, test_h5g, test_h5i, test_h5p, \
       test_h5r, test_h5s, test_h5t
import test_conv, test_utils

modules = [ test_h5,  test_h5a, test_h5d, test_h5f,
            test_h5g, test_h5i, test_h5p, test_h5r,
            test_h5s, test_h5t,
            test_conv, test_utils ]

def getsuite():
    """ Return a test suite containing all low-level tests """
    ldr = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module in modules:
        suite.addTests(ldr.loadTestsFromModule(module))
    return suite


