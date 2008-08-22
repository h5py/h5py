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
import test_h5a, test_h5d, test_h5f, \
       test_h5g, test_h5i, test_h5p, \
       test_h5s, test_h5t, test_h5, \
       test_highlevel, test_threads

from h5py import *

sections = {'h5a': test_h5a.TestH5A, 'h5d': test_h5d.TestH5D,
            'h5f': test_h5f.TestH5F, 'h5g': test_h5g.TestH5G,
            'h5i': test_h5i.TestH5I, 'h5p': test_h5p.TestH5P,
            'h5p.fcid': test_h5p.TestFCID, 'h5p.faid': test_h5p.TestFAID,
            'h5p.dcid': test_h5p.TestDCID, 'h5p.dxid': test_h5p.TestDXID,
            'h5s': test_h5s.TestH5S, 'h5t': test_h5t.TestH5T,
            'h5': test_h5.TestH5,
            'File': test_highlevel.TestFile,
            'Group': test_highlevel.TestGroup,
            'Dataset': test_highlevel.TestDataset,
            'threads': test_threads.TestThreads }

order = ('h5a', 'h5d', 'h5f', 'h5g', 'h5i', 'h5p', 'h5p.fcid', 'h5p.faid',
         'h5p.dcid', 'h5p.dxid', 'h5s', 'h5', 'File', 'Group', 'Dataset',
         'threads')

def buildsuite(cases):
    """ cases should be an iterable of TestCase subclasses """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return suite

def runtests(requests=None):
    if requests is None:
        requests = ()
    excluded = [x[1:] for x in requests if len(x) >=2 and x.find('-') == 0]
    included = [x for x in requests if x.find('-') != 0]

    cases = order if len(included) is 0 else included
    cases = [x for x in cases if x not in excluded]
    try:
        cases = [sections[x] for x in cases]
    except KeyError, e:
        raise RuntimeError('Test "%s" is unknown' % e.args[0])

    suite = buildsuite(cases)
    retval = unittest.TextTestRunner(verbosity=3).run(suite)
    print "=== Tested HDF5 %s (%s API) ===" % (h5.hdf5_version, h5.api_version)
    return retval.wasSuccessful()

def autotest():
    try:
        if not runtests():
            sys.exit(1)
    except:
        sys.exit(2)



