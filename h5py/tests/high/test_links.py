

from h5py import tests
import h5py

class TestClass(tests.HTest):

    def test_spath(self):
        """ (Links) Soft link path attribute """
        sl = h5py.SoftLink('/foo')
        self.assertEqual(sl.path, '/foo')

    def test_srepr(self):
        """ (Links) Soft link repr """
        sl = h5py.SoftLink('/foo')
        self.assertIsInstance(repr(sl), basestring)

    @tests.require(api=18)
    def test_epath(self):
        """ (Links) External path/file attributes """
        el = h5py.ExternalLink('foo.hdf5', '/foo')
        self.assertEqual(el.filename, 'foo.hdf5')
        self.assertEqual(el.path, '/foo')

    @tests.require(api=18)
    def test_erepr(self):
        """ (Links) External link repr() """
        el = h5py.ExternalLink('foo.hdf5','/foo')
        self.assertIsInstance(repr(el), basestring)

class Base(tests.HTest):

    def setUp(self):
        import tempfile
        self.name = tempfile.mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        self.f.close()
        import os
        os.unlink(self.name)

class TestSoft(Base):

    def test_create(self):
        """ (Links) Soft links creation """
        g = self.f.create_group('new')
        sl = h5py.SoftLink('/new')
        self.f['alias'] = sl
        g2 = self.f['alias']
        self.assertEqual(g, g2)
        self.assertEqual(self.f.get('alias', getclass=True, getlink=True), h5py.SoftLink)

    def test_exc(self):
        """ (Links) Dangling soft links raise KeyError """
        self.f['alias'] = h5py.SoftLink('new')
        self.assertRaises(KeyError, self.f.__getitem__, 'alias')



