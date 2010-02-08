from tempfile import mktemp

from h5py import tests
import h5py

class Base(tests.HTest):

    def setUp(self):
        self.name = mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        import os
        try:
            if self.f:
                self.f.close()
        finally:
            if self.name and os.path.exists(self.name):
                os.unlink(self.name)

class TestPropFile(Base):

    def test_file2(self):
        """ (HLObject) .file property on subclasses """
        g = self.f.create_group('foo')
        g2 = self.f.create_group('foo/bar')
        self.assertEqual(self.f, self.f.file)
        self.assertEqual(self.f, g.file)
        self.assertEqual(self.f, g2.file)

class TestProps(Base):

    @tests.require(api=18)
    def test_lcpl(self):
        """ (HLObject) lcpl """
        lcpl = self.f._lcpl
        self.assertIsInstance(lcpl, h5py.h5p.PropLCID)

    @tests.require(api=18)
    def test_lapl(self):
        """ (HLObject) lapl """
        lapl = self.f._lapl
        self.assertIsInstance(lapl, h5py.h5p.PropLAID)

class TestParent(Base):

    @tests.fixme("File object does not compare equal to Group '/'")
    def test_parent(self):
        """ (HLObject) .parent """
        self.assertEqual(self.f.parent, self.f)
        g = self.f.create_group('a')
        g2 = self.f.create_group('b')
        self.assertEqual(g2.parent, g)
        self.assertEqual(g.parent, self.f)






