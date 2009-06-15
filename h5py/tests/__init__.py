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

import h5py.tests
import unittest

mnames = [
'test_dataset',
'test_filters',
'test_h5a',
'test_h5d',
'test_h5f',
'test_h5g',
'test_h5i',
'test_h5p',
'test_h5',
'test_h5r',
'test_h5s',
'test_h5t',
'test_highlevel',
'test_slicing',
'test_threads',
'test_utils',
'test_vlen']


def runtests():

    ldr = unittest.TestLoader()
    suite = unittest.TestSuite()
    modules = [__import__('h5py.tests.'+x, fromlist=[h5py.tests]) for x in mnames]
    for m in modules:
        suite.addTests(ldr.loadTestsFromModule(m))

    runner = unittest.TextTestRunner()
    return runner.run(suite)

def autotest():
    try:
        if not runtests():
            sys.exit(17)
    except:
        sys.exit(2)



