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

config = h5py.h5.get_config()

def runtests():
    """ Run low and highlevel h5py tests.

    Result is a 2-tuple of TestResult objects
    """
    import low
    runner = unittest.TextTestRunner()
    return tuple(runner.run(suite) for suite in (low.getsuite(),))

def autotest():
    try:
        if not all(runtests()):
            sys.exit(17)
    except:
        sys.exit(2)

def requires(api=None):
    """ Decorator to enable/disable tests """
    def wrap(func):
        if api == 18 and not config.API_18: return None
        if api == 16 and not config.API_16: return None
        return func
    return wrap

def skip(func):
    """ Decorator to disable a test """
    return None

def datapath(name):
    """ Path to a data file shipped with the test suite """
    import os.path
    return os.path.join(os.path.dirname(__file__), 'data', name)

class FIDProxy(object):

    """
        Proxy object for low-level file IDs
    """

    def __init__(self, name=None):
        import tempfile
        import shutil
        import os.path as op
        from h5py import h5f, h5p

        plist = h5py.h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)

        targetname = tempfile.mktemp('.hdf5')

        if name is None:
            fid = h5f.create(targetname, h5f.ACC_TRUNC, fapl=plist)
        else:
            name = datapath(name)
            shutil.copy(name, targetname)
            fid = h5f.open(targetname, h5f.ACC_RDWR, fapl=plist)

        self.name = targetname
        self.fid = fid

    def erase(self):
        """ Ensure the FID is closed and remove the temporary file """
        import os
        while self.fid and h5py.h5i.get_ref(self.fid) > 0:
            h5py.h5i.dec_ref(self.fid)
        os.unlink(self.name)

class TestCasePlus(unittest.TestCase):

    """
        Slightly modified subclass of TestCase, which provides the following
        added functionality:

        1. assertArrayEqual(), to save us from having to remember to use
           numpy.all.  Also allows reasonable comparison of floating-point
           data.

        2. log() method, which allows useful information to be logged
           without resorting to gyrations with sys.stderr or the logging
           module.
    """

    EPSILON = 1e-5

    def log(self, msg):
        """ Print a message to an internal "log buffer", which is purged
            when a new test begins.
        """
        self._log += msg+'\n'

    def run(self, *args, **kwds):
        import warnings

        self._log = ""
        filters = warnings.filters
        warnings.simplefilter("ignore")
        try:
            unittest.TestCase.run(self, *args, **kwds)
        except Exception, e:
            if len(e.args) == 1 and isinstance(e.args[0], basestring):
                e.args = (e.args[0]+'\n'+self._log)
            elif len(e.args) == 2 and isinstance(e.args[1], basestring):
                e.args = (e.args[1]+'\n'+self._log)
            else:
                e.args += (self._log)
            raise
        finally:
            warnings.filters = filters

    def assertArrayEqual(self, dset, arr, message=None, precision=None):
        """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
        if precision is None:
            precision = EPSILON
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



