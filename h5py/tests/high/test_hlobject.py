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

class TestComparison(Base):

    def test_eq(self):
        """ (HLObject) __eq__ and __ne__ are opposite (files and groups) """
        g1 = self.f.create_group('a')
        g2 = self.f['a']
        g3 = self.f.create_group('b')
        f1 = self.f
        f2 = g1.file
        self.assert_(g1 == g2)
        self.assert_(not g1 != g2)
        self.assert_(g1 != g3)
        self.assert_(not g1 == g3)
        self.assert_(f1 == f2)
        self.assert_(not f1 != f2)
        
    def test_grp(self):
        """ (HLObject) File objects don't compare equal to root group """
        g = self.f['/']
        self.assert_(not g == self.f)
        self.assert_(g != self.f)

class TestProps(Base):

    def test_file2(self):
        """ (HLObject) .file """
        g = self.f.create_group('foo')
        g2 = self.f.create_group('foo/bar')
        self.assertEqual(self.f, self.f.file)
        self.assertEqual(self.f, g.file)
        self.assertEqual(self.f, g2.file)

    def test_parent(self):
        """ (HLObject) .parent """
        self.assertEqual(self.f.parent, self.f['/'])
        g = self.f.create_group('a')
        g2 = self.f.create_group('a/b')
        self.assertEqual(g2.parent, g)
        self.assertEqual(g.parent, self.f['/'])

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








