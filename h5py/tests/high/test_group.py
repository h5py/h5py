
import numpy as np
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

    """ Creating groups """

    def test_create(self):
        """ (Group) Create with create_group() """
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

class TestDel(GroupBase):

    """ Deleting objects """

    def test_del(self):
        """ (Group) del unlinks group """
        self.f.create_group('new')
        self.assert_('new' in self.f)
        del self.f['new']
        self.assert_('new' not in self.f)

    def test_del_exc(self):
        """ (Group) del raises KeyError for missing item """
        self.assertRaises(KeyError, self.f.__delitem__, 'new')

class TestSpecial(GroupBase):

    """ Misc HL object protocol """

    def test_repr(self):
        """ (Group) repr() returns string """
        g = self.f.create_group('new')
        self.assertIsInstance(repr(g), basestring)
        g.id._close()
        self.assertIsInstance(repr(g), basestring)

    def test_bool(self):
        """ (Group) nonzero() tracks object validity """
        g = self.f.create_group('new')
        self.assert_(g)
        g.id._close()
        self.assert_(not g)

class TestDataset(GroupBase):

    """ Creating datasets """

    def test_dataset(self):
        """ (Group) Create dataset via create_dataset """
        ds = self.f.create_dataset("Dataset", shape=(10,10), dtype='<i4')
        self.assertIsInstance(ds, h5py.Dataset)
        self.assert_("Dataset" in self.f)

class TestDict(GroupBase):

    """ Dictionary API """

    def setUp(self):
        GroupBase.setUp(self)
        self.groups = ('a','b','c','d')
        for x in self.groups:
            self.f.create_group(x)

    def test_len(self):
        """ (Group) len() tracks group contents """
        self.assertEqual(len(self.f), len(self.groups))
        self.f.create_group('e')
        self.assertEqual(len(self.f), len(self.groups)+1)

    def test_contains(self):
        """ (Group) __contains__ works for absolute and relative paths"""
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

class TestAutoCreate(GroupBase):

    """ __setitem__ for scalars, sequences & datatypes """

    def test_scalar(self):
        """ (Group) Store scalar -> rank-0 dataset """
        self.f['x'] = 42
        x = self.f['x']
        self.assertIsInstance(x, h5py.Dataset)
        self.assertEqual(x.shape, ())
        self.assertEqual(x[()], 42)

    def test_sequence(self):
        """ (Group) Store sequence -> dataset """
        self.f['x'] = [1,2,3,4,5]
        x = self.f['x']
        self.assertIsInstance(x, h5py.Dataset)
        self.assertEqual(x.shape, (5,))
        self.assertArrayEqual(x, np.array([1,2,3,4,5]))

    def test_fortran(self):
        """ (Group) Assign Fortran array """
        a = np.array([[1,2,3],[4,5,6]], order='f')
        self.f['x'] = a
        x = self.f['x']
        self.assertArrayEqual(x, a)

    def test_dtype(self):
        """ (Group) Store dtype -> named type """
        dt = np.dtype('f')
        self.f['x'] = dt
        x = self.f['x']
        self.assertIsInstance(x, h5py.Datatype)
        self.assertEqual(x.dtype, dt)

    def test_hardlink(self):
        """ (Group) Store HLObject -> hard link """
        g = self.f.create_group('foo')
        self.f['x'] = g
        x = self.f['x']
        self.assertIsInstance(x, h5py.Group)
        self.assertEqual(x, g)

    def test_exc(self):
        """ (Group) Assign with non-string key raises TypeError """
        self.assertRaises(TypeError, self.f.__setitem__, 42, 42)

    def test_exc1(self):
        """ (Group) Assign with existing key raises ValueError """
        self.f.create_group('foo')
        self.assertRaises(ValueError, self.f.__setitem__, 'foo', 42)

class TestVisit(GroupBase):

    def setUp(self):
        GroupBase.setUp(self)
        self.groups = ['grp1', 'grp1/sg1', 'grp1/sg2', 'grp2', 'grp2/sg1', 'grp2/sg1/ssg1']
        for x in self.groups:
            self.f.create_group(x)

    @tests.require(api=16)
    def test_visit_16(self):
        """ (Group) visit() on 1.6 raises NotImplementedError """
        l = []
        self.assertRaises(NotImplementedError, self.f.visit, l.append)

    @tests.require(api=18)
    def test_visit_18(self):
        """ (Group) visit() on 1.8 iterates over all names """
        l = []
        self.f.visit(l.append)
        self.assertEqualContents(l, self.groups)

    @tests.require(api=16)
    def test_visititems_16(self):
        """ (Group) visititems() on 1.6 raises NotImplementedError """
        l = []
        self.assertRaises(NotImplementedError, self.f.visititems, lambda x, y: l.append((x,y)))

    @tests.require(api=18)
    def test_visititems_18(self):
        """ (Group) visititems() on 1.8 iterates over all names, values """
        l = []
        comp = [(x, self.f[x]) for x in self.groups]
        self.f.visititems(lambda x, y: l.append((x,y)))
        self.assertEqualContents(comp, l)

    @tests.require(api=18)
    def test_short(self):
        """ (Group) visit(), visititems() honor short-circuit return value """
        x = self.f.visit(lambda x: x)
        self.assertEqual(x, self.groups[0])
        x = self.f.visititems(lambda x, y: (x,y))
        self.assertEqual(x, (self.groups[0], self.f[self.groups[0]]))


















