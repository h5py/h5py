

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

class TestExternal(Base):

    def setUp(self):
        Base.setUp(self)
        import tempfile
        self.ename = tempfile.mktemp()
        self.ef = h5py.File(self.ename, 'w')
        g = self.ef.create_group('external')
        self.ef.close()

    def tearDown(self):
        Base.tearDown(self)
        import os
        os.unlink(self.ename)

    @tests.require(api=18)
    def test_create_only(self):
        """ (Links) Create external link """
        self.f['ext'] = h5py.ExternalLink(self.ename, '/external')

    @tests.require(api=18)
    def test_create(self):
        """ (Links) Access external link """
        self.f['ext'] = h5py.ExternalLink(self.ename, '/external')
        g = self.f['ext']
        self.assert_(g)
        self.assertIsInstance(g, h5py.Group)

    @tests.require(api=18)
    def test_exc(self):
        """ (Links) Missing external link raises KeyError """
        self.f['ext'] = h5py.ExternalLink(self.ename, '/missing')
        self.assertRaises(KeyError, self.f.__getitem__, 'ext')

    @tests.require(api=18)
    def test_exc1(self):
        """ (Links) Missing external file raises IOError """
        self.f['ext'] = h5py.ExternalLink('misssing.hdf5', '/missing')
        self.assertRaises(IOError, self.f.__getitem__, 'ext')

    @tests.require(api=18)
    def test_file(self):
        """ (Links) File attribute works correctly on external links """
        self.f['ext'] = h5py.ExternalLink(self.ename, '/external')
        g = self.f['ext']
        self.assertNotEqual(g.file, self.f)













