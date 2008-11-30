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

import common
from common import HDF5TestCase
import  test_h5a, test_h5d, test_h5f,\
        test_h5g, test_h5i, test_h5p,\
        test_h5s, test_h5t, test_h5,\
        test_h5r,\
        test_highlevel, test_threads, test_utils

from h5py import *

sections = {'h5a': test_h5a.TestH5A, 'h5d': test_h5d.TestH5D,
            'h5f': test_h5f.TestH5F, 'h5g': test_h5g.TestH5G,
            'h5i': test_h5i.TestH5I, 'h5p': test_h5p.TestH5P,
            'h5p.fcid': test_h5p.TestFCID, 'h5p.faid': test_h5p.TestFAID,
            'h5p.dcid': test_h5p.TestDCID, 'h5p.dxid': test_h5p.TestDXID,
            'h5s': test_h5s.TestH5S, 'h5t': test_h5t.TestH5T,
            'h5r': test_h5r.TestH5R,
            'h5': test_h5.TestH5,
            'File': test_highlevel.TestFile,
            'Group': test_highlevel.TestGroup,
            'Dataset': test_highlevel.TestDataset,
            'threads': test_threads.TestThreads,
            'utils': test_utils.TestUtils }

def runtests(requests=None, verbosity=1):
    """Run unit tests

    Requests: iterable of test section names to run. Prefix with '-'
    to explicitly disable.  Example:
        ('h5a', 'h5g', 'File'), or ('-h5t', '-h5g')

    Verbosity: TextTestRunner verbosity level.  Level 4 prints additional
    h5py-specific information.
    """
    if requests is None:
        requests = ()
    excluded = [x[1:] for x in requests if len(x) >=2 and x.find('-') == 0]
    included = [x for x in requests if x.find('-') != 0]

    cases = tuple(sections) if len(included) is 0 else included
    cases = [x for x in cases if x not in excluded]
    try:
        cases = [sections[x] for x in cases]
    except KeyError, e:
        raise RuntimeError('Test "%s" is unknown' % e.args[0])

    HDF5TestCase.h5py_verbosity = verbosity
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in sorted(cases):
        suite.addTests(loader.loadTestsFromTestCase(case))

    retval = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    if verbosity >= 1:
        print "=== Tested HDF5 %s (%s API) ===" % (version.hdf5_version, version.api_version)

    return retval.wasSuccessful()

def autotest():
    try:
        if not runtests():
            sys.exit(17)
    except:
        sys.exit(2)



