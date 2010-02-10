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
import h5py
import numpy as np

config = h5py.h5.get_config()

def autotest():
    try:
        if not all(runtests()):
            sys.exit(17)
    except:
        sys.exit(2)

def fixme(desc=""):

    def wrap(func):
        print "FIXME: %s [%s]" % (func.__doc__, desc)
        return None
    return wrap

class _placeholder(object):
    pass

def require(condition=_placeholder, api=None, os=None, unicode=None):
    """ Decorator to enable/disable tests """
    import sys
    def haveunicode():
        import os.path
        try:
            os.path.exists(u'\u201a')
        except UnicodeError:
            return False
        return True
    def wrap(func):
        if condition is not _placeholder and not condition: return None
        if unicode and not haveunicode(): return None
        if api == 18 and not config.API_18: return None
        if api == 16 and config.API_18: return None
        if os == 'windows' and sys.platform != 'win32': return None
        if os == 'unix' and sys.platform == 'win32': return None
        return func
    return wrap

def skip(func):
    """ Decorator to disable a test """
    return None

def getpath(name):
    """ Path to a data file shipped with the test suite """
    import os.path
    return os.path.join(os.path.dirname(__file__), 'data', name)

def gettemp():
    """ Create a temporary file and return a 2-tuple (fid, name) """
    import tempfile
    name = tempfile.mktemp('.hdf5')
    plist = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    plist.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
    fid = h5py.h5f.create(name, h5py.h5f.ACC_TRUNC, fapl=plist)
    return fid, name


class HTest(unittest.TestCase):

    """
        Slightly modified subclass of TestCase, which provides the following
        added functionality:

        1. assertArrayEqual(), to save us from having to remember to use
           numpy.all.  Also allows reasonable comparison of floating-point
           data.

        2. Catches and suppresses warnings.
    """

    EPSILON = 1e-5

    def run(self, *args, **kwds):
        import warnings
        filters = warnings.filters
        warnings.simplefilter("ignore")
        objcount = h5py.h5f.get_obj_count()
        try:
            unittest.TestCase.run(self, *args, **kwds)
        finally:
            warnings.filters = filters
            newcount = h5py.h5f.get_obj_count()
            if newcount != objcount:
                print "WARNING: LEAKED %d IDs (total %d)" % (newcount-objcount, newcount)
                print h5py.h5f.get_obj_ids()
                if 0:
                    ids = h5py.h5f.get_obj_ids()
                    for id_ in ids:
                        if id_:
                            print "Closing %r" % id_
                        else:
                            print "Skipping %r" % id_
                        while id_ and h5py.h5i.get_ref(id_) > 0:
                            h5py.h5i.dec_ref(id_)

    def assertArrayEqual(self, dset, arr, message=None, precision=None):
        """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
        if precision is None:
            precision = self.EPSILON
        if message is None:
            message = ''
        else:
            message = ' (%s)' % message

        if np.isscalar(dset) or np.isscalar(arr):
            assert np.isscalar(dset) and np.isscalar(arr), 'Scalar/array mismatch ("%r" vs "%r")%s' % (dset, arr, message)
            assert dset - arr < precision, "Scalars differ by more than %.3f%s" % (precision, message)
            return

        assert dset.shape == arr.shape, "Shape mismatch (%s vs %s)%s" % (dset.shape, arr.shape, message)
        assert dset.dtype == arr.dtype, "Dtype mismatch (%s vs %s)%s" % (dset.dtype, arr.dtype, message)
        assert np.all(np.abs(dset[...] - arr[...]) < precision), "Arrays differ by more than %.3f%s" % (precision, message)

    def assertIsInstance(self, obj, typ):
        """ Check if obj is an instance of typ """
        if not isinstance(obj, typ):
            raise AssertionError("instance of %s not %s" % (type(obj).__name__, typ.__name__))

    def assertEqualContents(self, a, b):
        a = list(a)
        b = list(b)
        if not len(a) == len(b) and set(a) == set(b):
            raise AssertionError("contents don't match: %s vs %s" % (list(a), list(b)))

    def assertIsNone(self, what):

        if what is not None:
            raise AssertionError("%r is not None" % what)

def runtests(**kwds):
    """ Run low and highlevel h5py tests.

    Result is a TestResult object.  Keywords forwarded to TextTestRunner.
    """
    import os, fnmatch
    import h5py.tests
    import h5py.tests.low
    import h5py.tests.high

    packages = ['low','high']
    ldr = unittest.TestLoader()
    suite = unittest.TestSuite()

    thisdir = os.path.dirname(__file__)

    # new tests
    for p in packages:
        files = [x.partition('.py')[0] for x in os.listdir(os.path.join(thisdir, p)) if fnmatch.fnmatch(x, 'test_*.py')]
        modules = ['h5py.tests.%s.%s' % (p, m) for m in files]
        modules = [__import__(m, fromlist=[h5py.tests, h5py.tests.low, h5py.tests.high]) for m in modules]
        for m in modules:
            suite.addTests(ldr.loadTestsFromModule(m))

    # old tests
    files = [x.partition('.py')[0] for x in os.listdir(thisdir) if fnmatch.fnmatch(x, 'test_*.py')]
    modules = ['h5py.tests.%s' % m for m in files]
    modules = [__import__(m, fromlist=[h5py.tests, h5py.tests.low, h5py.tests.high]) for m in modules]
    for m in modules:
        suite.addTests(ldr.loadTestsFromModule(m))

    runner = unittest.TextTestRunner(**kwds)
    return runner.run(suite)

def report():
    """ Generate a test report containing system parameters and test results.

    Return is a 3-tuple sysinfo, stdout, stderr.
    """
    import cStringIO
    import sys
    import os
    import h5py
    import time
    import numpy
    import platform

    plat = ['platform', 'python_version', 'python_compiler']
    info = """\
Time:   %(time)s
h5py:   %(vers)s
HDF5:   %(hvers)s
API:    %(hapi)s
NumPy:  %(numpy)s
"""
    info %= {'time': time.asctime(), 'vers': h5py.version.version,
             'hvers': h5py.version.hdf5_version, 'hapi': h5py.version.api_version,
             'numpy': numpy.version.version}
    info += "\n".join("%s: %s" % (x, getattr(platform, x)()) for x in plat)

    nso, nse = cStringIO.StringIO(), cStringIO.StringIO()
    oso, ose = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = nso, nse
        result = runtests(stream=nse, verbosity=3)
    finally:
        sys.stdout, sys.stderr, = oso, ose
    nso.seek(0)
    nse.seek(0)

    return info, nso.read(), nse.read()








