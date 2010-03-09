
from __future__ import with_statement

from tempfile import mktemp

from h5py import tests
import h5py

class FileBase(tests.HTest):

    def setUp(self):
        self.f = None
        self.name = None

    def tearDown(self):
        import os
        try:
            if self.f:
                self.f.close()
        finally:
            if self.name and os.path.exists(self.name):
                os.unlink(self.name)

class TestOpening(FileBase):

    def test_w(self):
        """ (File) Create/truncate """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')
        self.assert_(self.f)
        self.f.create_group('g')
        self.f.close()
        self.f = h5py.File(self.name, 'w')
        self.assert_('g' not in self.f)

    def test_wm(self):
        """ (File) Create/exclusive """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w-')
        self.assert_(self.f)
        self.f.close()
        self.assertRaises(IOError, h5py.File, self.name, 'w-')

    def test_a(self):
        """ (File) Append """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'a')
        self.assert_(self.f)
        self.f.create_group('g')
        self.f.close()
        self.f = h5py.File(self.name, 'a')
        self.assert_('g' in self.f)

    @tests.require(h5py.version.hdf5_version > (1,8,2))
    def test_r(self):
        """ (File) Readonly """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')
        self.f.create_group('g')
        self.f.close()
        self.f = h5py.File(self.name, 'r')
        self.assert_('g' in self.f)
        self.assertRaises(IOError, self.f.create_group, 'gg')

    def test_rp(self):
        """ (File) Readwrite """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')
        self.f.create_group('g')
        self.f.close()
        self.f = h5py.File(self.name, 'r+')
        self.assert_('g' in self.f)
        self.f.create_group('gg')
        self.assert_('gg' in self.f)

    def test_exc_1(self):
        """ (File) Missing file causes IOError with r, r+ """
        name = mktemp()
        self.assertRaises(IOError, h5py.File, name, 'r')
        self.assertRaises(IOError, h5py.File, name, 'r+')

    def test_context(self):
        """ (File) Using File object as context manager """
        self.name = mktemp()
        with h5py.File(self.name, 'w') as self.f:
            self.assert_(self.f)
        self.assert_(not self.f)

class TestDrivers(FileBase):

    @tests.require(os='unix')
    def test_stdio(self):
        """ (File) Create with stdio driver """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w', driver='stdio')
        self.assert_(self.f)
        self.assertEqual(self.f.driver, 'stdio')

    @tests.require(os='unix')
    def test_sec2(self):
        """ (File) Create with sec2 driver """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w', driver='sec2')
        self.assert_(self.f)
        self.assertEqual(self.f.driver, 'sec2')

class TestCore(FileBase):

    def test_create(self):
        """ (File) Create with core driver """
        f = h5py.File('a', 'w', driver='core', backing_store=False)
        self.assert_(f)
        self.assertEqual(f.driver, 'core')
        f.close()

    @tests.require(api=18)
    def test_open(self):
        """ (File) Open with core driver on 1.8 """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')
        self.f.create_group('g')
        self.f.close()
        f = h5py.File(self.name, 'r', driver='core')
        self.assert_(f)
        self.assert_('g' in f)
        f.close()

    def test_backing(self):
        """ (File) Backing store for core driver """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w', driver='core', backing_store=True)
        self.f.create_group('g')
        self.f.close()
        self.f = h5py.File(self.name, 'r')
        self.assert_('g' in self.f)

    def test_blocksize(self):
        """ (File) Block size argument for core driver """
        self.f = h5py.File('a', 'w', driver='core', block_size=1024, backing_store=False)
        self.assert_(self.f)

class TestOps(tests.HTest):

    def setUp(self):
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        if self.f:
            self.f.close()
        import os
        os.unlink(self.name)

    def test_close(self):
        """ (File) File close() affects __nonzero__() """
        self.assert_(self.f)
        self.f.close()
        self.assert_(not self.f)

    def test_close_multi(self):
        """ (File) File close() affects all copies of identifier """
        g = self.f.create_group('a')
        f2 = g.file
        self.assert_(f2 is not self.f)
        self.f.close()
        self.assert_(not self.f)
        self.assert_(not f2)

    def test_exc(self):
        """ (File) I/O on closed file results in ValueError """
        self.f.close()
        self.assertRaises(ValueError, self.f.create_group, 'g')

    def test_flush(self):
        """ (File) Flush method """
        self.f.flush()

    def test_repr(self):
        """ (File) repr() """
        self.assert_(isinstance(repr(self.f), basestring))
        self.f.close()
        self.assert_(isinstance(repr(self.f), basestring))

    def test_fileprop(self):
        """ (File) File property """
        self.assert_(self.f.file is self.f)

    def test_filename(self):
        """ (File) Filename property """
        self.assertEqual(self.f.filename, self.name)
        self.assert_(isinstance(self.f.filename, str))

class TestUnicode(FileBase):

    @tests.require(unicode=True)
    def test_unicode(self):
        """ (File) Unicode filenames """
        self.name = mktemp(u'\u201a')
        self.f = h5py.File(self.name, 'w')
        self.assertEqual(self.f.filename, self.name)
        self.assert_(isinstance(self.f.filename, unicode))

class TestMode(FileBase):

    def test_mode(self):
        """ (File) Mode works properly with .file attribute """
        self.name = mktemp()
        self.f = h5py.File(self.name, 'a')
        g = self.f.create_group('g')
        self.assertEqual(g.file.mode, self.f.mode)
        self.assertEqual(self.f.mode, 'a')
        




