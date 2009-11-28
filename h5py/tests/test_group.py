from __future__ import with_statement

"""
    Test highlevel Group object behavior
"""
import numpy as np

import h5py
import warnings
from common import TestCasePlus, api_16, api_18, res

SHAPES = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]

class GroupBase(TestCasePlus):

    """
        Base class to handle Group setup/teardown, and some shared logic.
    """

    def setUp(self):
        self.f = h5py.File(res.get_name(), 'w')

    def tearDown(self):
        res.clear()

    def assert_equal_contents(self, a, b):
        """ Check if two iterables contain the same elements, regardless of
            order.
        """
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(a), len(b))

class TestInit(GroupBase):

    """ Group creation """

    def test_Group_init(self):
        """ Group constructor """

        grp = h5py.Group(self.f, "NewGroup", create=True)
        self.assert_("NewGroup" in self.f)
        grp2 = h5py.Group(self.f, "NewGroup")

        self.assertEqual(grp.name, "/NewGroup")

    def test_Group_create_group(self):

        grp = self.f.create_group("NewGroup")
        self.assert_("NewGroup" in self.f)
        self.assertRaises(ValueError, self.f.create_group, "NewGroup")

    def test_Group_create_dataset(self):

        ds = self.f.create_dataset("Dataset", shape=(10,10), dtype='<i4')
        self.assert_(isinstance(ds, h5py.Dataset))
        self.assert_("Dataset" in self.f)

class TestSpecial(GroupBase):

    """ Special methods """

    subgroups = ["Name1", " Name 1231987&*@&^*&#W  2  \t\t ",
                 "name3", "14", "!"]

    def setUp(self):
        GroupBase.setUp(self)
        for name in self.subgroups:
            self.f.create_group(name)

    def test_len(self):
        self.assertEqual(len(self.f), len(self.subgroups))

    def test_contains(self):
        for name in self.subgroups:
            self.assert_(name in self.f)
        self.assert_("missing" not in self.f)

    def test_iter(self):
        self.assert_equal_contents(self.f, self.subgroups)

    def test_dictcompat(self):

        # Old style -- now deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assert_equal_contents(self.f.listnames(), self.subgroups)
            self.assert_equal_contents(self.f.listobjects(), [self.f[x] for x in self.subgroups])
            self.assert_equal_contents(self.f.listitems(), [(x, self.f[x]) for x in self.subgroups])
            self.assert_equal_contents(list(self.f.iternames()), self.subgroups)
            self.assert_equal_contents(list(self.f.iterobjects()), [self.f[x] for x in self.subgroups])

        # New style
        self.assert_equal_contents(self.f.keys(), self.subgroups)
        self.assert_equal_contents(self.f.values(), [self.f[x] for x in self.subgroups])
        self.assert_equal_contents(self.f.items(), [(x, self.f[x]) for x in self.subgroups])
        self.assert_equal_contents(list(self.f.iteritems()), [(x, self.f[x]) for x in self.subgroups])


    def test_del(self):
        names = list(self.subgroups)
        for name in self.subgroups:
            names.remove(name)
            del self.f[name]
            self.assert_equal_contents(self.f, names)

    def test_str_repr(self):
        g = self.f.create_group("Foobar")
        s = str(g)
        r = repr(g)
        self.assert_(isinstance(s, str),s)
        self.assert_(isinstance(r, str),r)
        self.assert_(r.startswith('<') and r.endswith('>'),r)

class TestSetItem(GroupBase):

    """ Behavior of Group.__setitem__ """

    def test_ndarray(self):
        """ Assignment of an ndarray to a Group object """

        shapes = [(), (1,), (10,10)]
        types = ['i', 'f']#, [('a','i'), ('b','f')]]
        types = [np.dtype(t) for t in types]

        for shape in shapes:
            for dt in types:
                msg = "Assign ndarray %s %s" % (dt, shape)

                arr = np.ones(shape, dtype=dt)
                self.f['DS'] = arr
                dset = self.f['DS']
                self.assertEqual(dset.shape, arr.shape)
                self.assertEqual(dset.dtype, arr.dtype)

                self.assertArrayEqual(dset[()], arr[()], msg)

                del self.f['DS']

    def test_dtype(self):
        """ Assignment of a dtype to a Group object """

        types = ['i', 'f', [('a','i'), ('b','f')]]
        types = [np.dtype(t) for t in types]

        for dt in types:
            msg = "Assign dtype %s" % (dt,)

            self.f['TYPE'] = dt
            htype = self.f['TYPE']
            self.assert_(isinstance(htype, h5py.Datatype), msg)
            self.assertEqual(htype.dtype, dt)
            
            del self.f['TYPE']

    def test_hardlink(self):
        """ Hard-linking by direct assignment to Group """

        grp1 = self.f.create_group('grp1')
        self.f['grp2'] = grp1
        grp2 = self.f['grp2']

        self.assert_(grp1 is not grp2)
        self.assert_(grp1 == grp2)
        self.assert_(hash(grp1) == hash(grp2))

    def test_array_autocreate(self):
        """ Auto-creation of dataset from sequence """

        seq = [1,-42,2,3,4,5,10]
        arr = np.array(seq)

        self.f['DS'] = seq
        self.assert_(np.all(self.f['DS'][...] == arr))

        # test scalar -> 0-d dataset
        self.f["DS_SC"] = 42
        harr = self.f["DS_SC"]
        self.assert_(isinstance(harr, Dataset))
        self.assertEqual(harr.shape, ())
        self.assertEqual(harr.value, 42)
    
        # test assignment of out-of-order arrays
        arr = np.array(numpy.arange(100).reshape((10,10)), order='F')
        self.f['FORTRAN'] = arr
        dset = self.f['FORTRAN']
        self.assert_(np.all(dset[:] == arr))
        self.assert_(dset[:].flags['C_CONTIGUOUS'])

    def test_exceptions(self):
        """ Exceptions from __setitem__ """

        self.f.create_group('grp')
        self.assertRaises(ValueError, self.f.create_group, 'grp')

def TestGetItem(GroupBase):

    def test_get(self):

        grp = self.f.create_group("grp")
        dset = self.f.create_dataset("ds", (1,), 'f')
        dtype = np.dtype('f')
        self.f['type'] = dtype
        htype = self.f['type']

        self.assert_(isinstance(self.f['grp'], h5py.Group))
        self.assert_(self.f['grp'] == grp)

        self.assert_(isinstance(self.f['ds'], h5py.Dataset))
        self.assert_(self.f['ds'] == dset)

        self.assert_(isinstance(self.f['type'], h5py.Datatype))
        self.assert_(self.f['type'].dtype == dtype)

    def test_exceptions(self):

        self.assertRaises(TypeError, self.f,__getitem__, 42)
        self.assertRaises(KeyError, self.f.__getitem__, 'missing')

## -----

from h5py import Group, Dataset, Datatype, File
import numpy

class TestOther(GroupBase):

    def test_require(self):

        grp = self.f.require_group('foo')
        self.assert_(isinstance(grp, Group))
        self.assert_('foo' in self.f)

        grp2 = self.f.require_group('foo')
        self.assert_(grp == grp2)
        self.assert_(hash(grp) == hash(grp2))

        dset = self.f.require_dataset('bar', (10,10), '<i4')
        self.assert_(isinstance(dset, Dataset))
        self.assert_('bar' in self.f)

        dset2 = self.f.require_dataset('bar', (10,10), '<i4')
        self.assert_(dset == dset2)
        self.assert_(hash(dset) == hash(dset2))

        self.assertRaises(TypeError, self.f.require_group, 'bar')
        self.assertRaises(TypeError, self.f.require_dataset, 'foo', (10,10), '<i4')

        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,11), '<i4')
        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,10), '<c8')
        self.assertRaises(TypeError, self.f.require_dataset, 'bar', (10,10), '<i1', exact=True)

        self.f.require_dataset('bar', (10,10), '<i1')

    @api_16
    def test_copy_16(self):

        self.f.create_group('foo')
        self.assertRaises(NotImplementedError, self.f.copy, 'foo', 'bar')

    @api_18
    def test_copy_18(self):

        self.f.create_group('foo')
        self.f.create_group('foo/bar')

        self.f.copy('foo', 'new')
        self.assert_('new' in self.f)
        self.assert_('new/bar' in self.f)

    @api_16
    def test_visit_16(self):

        for x in ['grp1','grp2']:
            self.f.create_group(x)

        grplist = []
        self.assertRaises(NotImplementedError, self.f.visit, grplist.append)

        self.assertRaises(NotImplementedError, self.f.visititems, lambda x,y: grplist.append((x,y)))

    @api_18
    def test_visit_18(self):

        groups = ['grp1', 'grp1/sg1', 'grp1/sg2', 'grp2', 'grp2/sg1', 'grp2/sg1/ssg1']

        for x in groups:
            self.f.create_group(x)

        group_visit = []
        self.f.visit(group_visit.append)

        self.assert_equal_contents(groups, group_visit)

        grp_items = [(x, self.f[x]) for x in groups]

        group_visit = []
        self.f.visititems(lambda x, y: group_visit.append((x,y)))
        
        self.assert_equal_contents(grp_items, group_visit)

        # Test short-circuit return

        group_visit = []
        def visitor(name, obj=None):
            group_visit.append(name)
            if name.find('grp2/sg1') >= 0:
                return name
            return None

        result = self.f.visit(visitor)
        self.assert_(result.find('grp2/sg1') >= 0)
        self.assert_(not any(x.find('grp2/sg1/ssg1') >= 0 for x in group_visit))

        del group_visit[:]

        result = self.f.visititems(visitor)
        self.assert_(result.find('grp2/sg1') >= 0)
        self.assert_(not any(x.find('grp2/sg1/ssg1') >= 0 for x in group_visit))
