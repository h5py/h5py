try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import tempfile, shutil, os
from h5py import File

class TestFileID(ut.TestCase):
    def test_descriptor_core(self):
        with File('TestFileID.test_descriptor_core', driver='core', backing_store=False) as f:
            with self.assertRaises(NotImplementedError):
                f.id.get_descriptor()

    def test_descriptor_sec2(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.test_descriptor_sec2')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, driver='sec2') as f:
                descriptor = f.id.get_descriptor()
                assert descriptor != 0
                # This part of the test only works on posix systems
                # TODO: find the analogous code for Windows (and test it).
                dn_fd = '/proc/%i/fd' % os.getpid()
                if os.path.isdir(dn_fd):
                    fn_h5_check = os.readlink('%s/%i' % (dn_fd, descriptor))
                    assert fn_h5_check == fn_h5
        finally:
            shutil.rmtree(dn_tmp)
