
from tempfile import mktemp
from h5py import tests
import h5py

class GroupBase(tests.HTest):

    def setUp(self):
        import tempfile
        self.name = tempfile.mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        import os
        self.f.close()
        os.unlink(self.name)

class TestCreate(GroupBase):

    def test_create(self):
        """ (Group) Create group """
        self.f.create_group('new')
        self.assert_('new' in self.f)
        g = self.f['new']
        self.assertIsInstance(g, h5py.Group)
        self.assertEqual(g.name, '/new')

    def test_conflict(self):
        """ (Group) Create with existing name raises ValueError """
        self.f.create_group('new')
        self.assertRaises(ValueError, self.f.create_group, 'new')

    def test_require(self):
        """ (Group) Create with require_group() """
        g = self.f.require_group('new')
        self.assertIsInstance(g, h5py.Group)
        self.assert_('new' in self.f)
        g2 = self.f.require_group('new')
        self.assertEqual(g, g2)

    def test_require_exc(self):
        """ (Group) require_group() raises TypeError with incompatible object """
        ds = self.f.create_dataset('new', (1,), 'f')
        self.assertRaises(TypeError, self.f.require_group, 'new')

    def test_del(self):
        """ (Group) del """
        self.f.create_group('new')
        self.assert_('new' in self.f)
        del self.f['new']
        self.assert_('new' not in self.f)

    @tests.fixme
    def test_del_exc(self):
        """ (Group) del raises KeyError for missing item """
        self.assertRaises(KeyError, self.f.__delitem__, 'new')

    def test_repr(self):
        """ (Group) repr() """
        g = self.f.create_group('new')
        repr(g)
        g.id._close()
        repr(g)

    def test_bool(self):
        """ (Group) nonzero() """
        g = self.f.create_group('new')
        self.assert_(g)
        g.id._close()
        self.assert_(not g)

class TestDict2(GroupBase):

    def setUp(self):
        GroupBase.setUp(self)
        self.groups = ('a','b','c','d')
        for x in self.groups:
            self.f.create_group(x)

    def test_len(self):
        """ (Group) len() """
        self.assertEqual(len(self.f), len(self.groups))
        self.f.create_group('e')
        self.assertEqual(len(self.f), len(self.groups)+1)

    def test_contains(self):
        """ (Group) __contains__ for absolute and relative paths"""
        self.assert_(not 'new' in self.f)
        g = self.f.create_group('new')
        self.assert_('new' in self.f)
        self.f.create_group('new/subgroup')
        self.assert_('subgroup' in g)
        self.assert_('/new/subgroup' in g)

    def test_iter(self):
        """ (Group) Iteration yields member names"""
        self.assertEqualContents(self.groups, self.f)

    def test_old(self):
        """ (Group) Deprecated dict interface """
        self.assertEqualContents(self.f.listnames(), self.groups)
        self.assertEqualContents(self.f.listobjects(), [self.f[x] for x in self.groups])
        self.assertEqualContents(self.f.listitems(), [(x, self.f[x]) for x in self.groups])
        self.assertEqualContents(list(self.f.iternames()), self.groups)
        self.assertEqualContents(list(self.f.iterobjects()), [self.f[x] for x in self.groups])

    def test_new(self):
        """ (Group) New dict interface """
        self.assertIsInstance(self.f.keys(), list)
        self.assertEqualContents(self.f.keys(), self.groups)

        self.assertIsInstance(self.f.values(), list)
        self.assertEqualContents(self.f.values(), [self.f[x] for x in self.groups])

        self.assertEqualContents(self.f.iteritems(), [(x,self.f[x]) for x in self.groups])
        self.assertEqualContents(self.f.iterkeys(), self.f)
        self.assertEqualContents(self.f.itervalues(), [self.f[x] for x in self.groups])



