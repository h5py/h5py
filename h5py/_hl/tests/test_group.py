
"""
    Group test module.

    Tests all methods and properties of Group objects, with the following
    exceptions:

    1. Method create_dataset is tested in module test_dataset
"""

import numpy as np
import sys

from .common import ut, TestCase
import h5py
from h5py.highlevel import File, Group, SoftLink, HardLink, ExternalLink
from h5py.highlevel import Dataset, Datatype
from h5py import h5t

class BaseGroup(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestRepr(BaseGroup):

    """
        Feature: repr() works sensibly on Group objects
    """

    def test_repr(self):
        """ repr() works on Group objects """
        g = self.f.create_group('foo')
        self.assertIsInstance(g, basestring)
        self.f.close()
        self.assertIsInstance(g, basestring)

class TestCreate(BaseGroup):

    """
        Feature: New groups can be created via .create_group method
    """

    def test_create(self):
        """ Simple .create_group call """
        grp = self.f.create_group('foo')
        self.assertIsInstance(grp, Group)

    def test_create_intermediate(self):
        """ Intermediate groups can be created automatically """
        grp = self.f.create_group('foo/bar/baz')
        self.assertEqual(grp.name, '/foo/bar/baz')

    def test_create_exception(self):
        """ Name conflict causes group creation to fail with ValueError """
        self.f.create_group('foo')
        with self.assertRaises(ValueError):
            self.f.create_group('foo')

    def test_unicode(self):
        """ Unicode names are correctly stored """
        name = u"/Name\u4500"
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_UTF8)

    def test_unicode_default(self):
        """ Unicode names convertible to ASCII are stored as ASCII (issue 239)
        """
        name = u"/Hello, this is a name"
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_ASCII)

    def test_appropriate_low_level_id(self):
        " Binding a group to a non-group identifier fails with ValueError "
        dset = self.f.create_dataset('foo', [1])
        with self.assertRaises(ValueError):
            Group(dset.id)

class TestDatasetAssignment(BaseGroup):

    """
        Feature: Datasets can be created by direct assignment of data
    """

    def test_ndarray(self):
        """ Dataset auto-creation by direct assignment """
        data = np.ones((4,4),dtype='f')
        self.f['a'] = data
        self.assertIsInstance(self.f['a'], Dataset)
        self.assertArrayEqual(self.f['a'][...], data)

class TestDtypeAssignment(BaseGroup):

    """
        Feature: Named types can be created by direct assignment of dtypes
    """

    def test_dtype(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        self.f['a'] = dtype
        self.assertIsInstance(self.f['a'], Datatype)
        self.assertEqual(self.f['a'].dtype, dtype)

class TestRequire(BaseGroup):

    """
        Feature: Groups can be auto-created, or opened via .require_group
    """

    def test_open_existing(self):
        """ Existing group is opened and returned """
        grp = self.f.create_group('foo')
        grp2 = self.f.require_group('foo')
        self.assertEqual(grp, grp2)

    def test_create(self):
        """ Group is created if it doesn't exist """
        grp = self.f.require_group('foo')
        self.assertIsInstance(grp, Group)
        self.assertEqual(grp.name, '/foo')

    def test_require_exception(self):
        """ Opening conflicting object results in TypeError """
        self.f.create_dataset('foo', (1,), 'f')
        with self.assertRaises(TypeError):
            self.f.require_group('foo')

class TestDelete(BaseGroup):

    """
        Feature: Objects can be unlinked via "del" operator
    """

    def test_delete(self):
        """ Object deletion via "del" """
        self.f.create_group('foo')
        self.assertIn('foo', self.f)
        del self.f['foo']
        self.assertNotIn('foo', self.f)

    def test_nonexisting(self):
        """ Deleting non-existent object raises KeyError """
        with self.assertRaises(KeyError):
            del self.f['foo']

    def test_readonly_delete_exception(self):
        """ Deleting object in readonly file raises KeyError """
        # Note: it is impossible to restore the old behavior (ValueError)
        # without breaking the above test (non-existing objects)
        fname = self.mktemp()
        hfile = File(fname,'w')
        try:
            hfile.create_group('foo')
        finally:
            hfile.close()

        hfile = File(fname, 'r')
        try:
            with self.assertRaises(KeyError):
                del hfile['foo']
        finally:
            hfile.close()
            
class TestOpen(BaseGroup):

    """
        Feature: Objects can be opened via indexing syntax obj[name]
    """

    def test_open(self):
        """ Simple obj[name] opening """
        grp = self.f.create_group('foo')
        grp2 = self.f['foo']
        grp3 = self.f['/foo']
        self.assertEqual(grp, grp2)
        self.assertEqual(grp, grp3)

    def test_nonexistent(self):
        """ Opening missing objects raises KeyError """
        with self.assertRaises(KeyError):
            self.f['foo']

    def test_reference(self):
        """ Objects can be opened by HDF5 object reference """
        grp = self.f.create_group('foo')
        grp2 = self.f[grp.ref]
        self.assertEqual(grp2, grp)

    def test_reference_numpyobj(self):
        """ Object can be opened by numpy.object_ containing object ref

        Test for issue 181, issue 202.
        """
        g = self.f.create_group('test')

        rtype = h5py.special_dtype(ref=h5py.Reference)
        dt = np.dtype([('a', 'i'),('b',rtype)])
        dset = self.f.create_dataset('test_dset', (1,), dt)

        dset[0] =(42,g.ref)
        data = dset[0]
        self.assertEqual(self.f[data[1]], g)

    # TODO: check that regionrefs also work with __getitem__

class TestRepr(BaseGroup):

    """
        Feature: Opened and closed groups provide a useful __repr__ string
    """

    def test_repr(self):
        """ Opened and closed groups provide a useful __repr__ string """
        g = self.f.create_group('foo')
        self.assertIsInstance(repr(g), basestring)
        g.id._close()
        self.assertIsInstance(repr(g), basestring)

class BaseMapping(BaseGroup):

    """
        Base class for mapping tests
    """
    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ('a','b','c','d')
        for x in self.groups:
            self.f.create_group(x)

    def tearDown(self):
        if self.f:
            self.f.close()

class TestLen(BaseMapping):

    """
        Feature: The Python len() function returns the number of groups
    """

    def test_len(self):
        """ len() returns number of group members """
        self.assertEqual(len(self.f), len(self.groups))
        self.f.create_group('e')
        self.assertEqual(len(self.f), len(self.groups)+1)

    def test_exc(self):
        """ len() on closed group gives ValueError """
        self.f.close()
        with self.assertRaises(ValueError):
            len(self.f)

class TestContains(BaseMapping):

    """
        Feature: The Python "in" builtin tests for containership
    """

    def test_contains(self):
        """ "in" builtin works for containership """
        self.assertIn('a', self.f)
        self.assertNotIn('mongoose', self.f)

    def test_exc(self):
        """ "in" on closed group returns False (see also issue 174) """
        self.f.close()
        self.assertFalse('a' in self.f)

class TestIter(BaseMapping):

    """
        Feature: You can iterate over group members via "for x in y", etc.
    """

    def test_iter(self):
        """ "for x in y" iteration """
        lst = [x for x in self.f]
        self.assertSameElements(lst, self.groups)

    def test_iter_zero(self):
        """ Iteration works properly for the case with no group members """
        hfile = File(self.mktemp(), 'w')
        try:
            lst = [x for x in hfile]
            self.assertEqual(lst, [])
        finally:
            hfile.close()
            
@ut.skipIf(sys.version_info[0] != 2, "Py2")
class TestPy2Dict(BaseMapping):

    """
        Feature: Standard Python 2 .keys, .values, etc. methods are available
    """

    def test_keys(self):
        """ .keys method """
        self.assertIsInstance(self.f.keys(), list)
        self.assertSameElements(self.f.keys(), self.groups)

    def test_values(self):
        """ .values method """
        self.assertIsInstance(self.f.values(), list)
        self.assertSameElements(self.f.values(), [self.f[x] for x in self.groups])

    def test_items(self):
        """ .items method """
        self.assertIsInstance(self.f.items(), list)
        self.assertSameElements(self.f.items(),
            [(x, self.f[x]) for x in self.groups])

    def test_iterkeys(self):
        """ .iterkeys method """
        self.assertSameElements([x for x in self.f.iterkeys()], self.groups)

    def test_itervalues(self):
        """ .itervalues method """
        self.assertSameElements([x for x in self.f.itervalues()],
            [self.f[x] for x in self.groups])

    def test_iteritems(self):
        """ .iteritems method """
        self.assertSameElements([x for x in self.f.iteritems()],
            [(x, self.f[x]) for x in self.groups])

@ut.skipIf(sys.version_info[0] != 3, "Py3")
class TestPy3Dict(BaseMapping):

    def test_keys(self):
        """ .keys provides a key view """
        kv = getattr(self.f, 'keys')()
        self.assertSameElements(list(kv), self.groups)
        for x in self.groups:
            self.assertIn(x, kv)
        self.assertEqual(len(kv), len(self.groups))

    def test_values(self):
        """ .values provides a value view """
        vv = getattr(self.f, 'values')()
        self.assertSameElements(list(vv), [self.f[x] for x in self.groups])
        self.assertEqual(len(vv), len(self.groups))
        with self.assertRaises(TypeError):
            b'x' in vv

    def test_items(self):
        """ .items provides an item view """
        iv = getattr(self.f, 'items')()
        self.assertSameElements(list(iv), [(x,self.f[x]) for x in self.groups])
        self.assertEqual(len(iv), len(self.groups))
        for x in self.groups:
            self.assertIn((x, self.f[x]), iv)

class TestGet(BaseGroup):

    """
        Feature: The .get method allows access to objects and metadata
    """

    def test_get_default(self):
        """ Object is returned, or default if it doesn't exist """
        default = object()
        out = self.f.get('mongoose', default)
        self.assertIs(out, default)

        grp = self.f.create_group('a')
        out = self.f.get('a')
        self.assertEqual(out, grp)

    def test_get_class(self):
        """ Object class is returned with getclass option """
        self.f.create_group('foo')
        out = self.f.get('foo', getclass=True)
        self.assertEqual(out, Group)

        self.f.create_dataset('bar', (4,))
        out = self.f.get('bar', getclass=True)
        self.assertEqual(out, Dataset)

        self.f['baz'] = np.dtype('|S10')
        out = self.f.get('baz', getclass=True)
        self.assertEqual(out, Datatype)

    def test_get_link_class(self):
        """ Get link classes """
        default = object()

        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el

        out_hl = self.f.get('hard', default, getlink=True, getclass=True)
        out_sl = self.f.get('soft', default, getlink=True, getclass=True)
        out_el = self.f.get('external', default, getlink=True, getclass=True)

        self.assertEqual(out_hl, HardLink)
        self.assertEqual(out_sl, SoftLink)
        self.assertEqual(out_el, ExternalLink)

    def test_get_link(self):
        """ Get link values """
        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el

        out_hl = self.f.get('hard', getlink=True)
        out_sl = self.f.get('soft', getlink=True)
        out_el = self.f.get('external', getlink=True)

        #TODO: redo with SoftLink/ExternalLink built-in equality
        self.assertIsInstance(out_hl, HardLink)
        self.assertIsInstance(out_sl, SoftLink)
        self.assertEqual(out_sl._path, sl._path)
        self.assertIsInstance(out_el, ExternalLink)
        self.assertEqual(out_el._path, el._path)
        self.assertEqual(out_el._filename, el._filename)

class TestVisit(TestCase):

    """
        Feature: The .visit and .visititems methods allow iterative access to
        group and subgroup members
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = [
            'grp1', 'grp1/sg1', 'grp1/sg2', 'grp2', 'grp2/sg1', 'grp2/sg1/ssg1'
            ]
        for x in self.groups:
            self.f.create_group(x)

    def tearDown(self):
        self.f.close()

    def test_visit(self):
        """ All subgroups are visited """
        l = []
        self.f.visit(l.append)
        self.assertSameElements(l, self.groups)

    def test_visititems(self):
        """ All subgroups and contents are visited """
        l = []
        comp = [(x, self.f[x]) for x in self.groups]
        self.f.visititems(lambda x, y: l.append((x,y)))
        self.assertSameElements(comp, l)

    def test_bailout(self):
        """ Returning a non-None value immediately aborts iteration """
        x = self.f.visit(lambda x: x)
        self.assertEqual(x, self.groups[0])
        x = self.f.visititems(lambda x, y: (x,y))
        self.assertEqual(x, (self.groups[0], self.f[self.groups[0]]))

class TestSoftLinks(BaseGroup):

    """
        Feature: Create and manage soft links with the high-level interface
    """
    
    def test_spath(self):
        """ SoftLink path attribute """
        sl = SoftLink('/foo')
        self.assertEqual(sl.path, '/foo')

    def test_srepr(self):
        """ SoftLink path repr """
        sl = SoftLink('/foo')
        self.assertIsInstance(repr(sl), basestring)

    def test_create(self):
        """ Create new soft link by assignment """
        g = self.f.create_group('new')
        sl = SoftLink('/new')
        self.f['alias'] = sl
        g2 = self.f['alias']
        self.assertEqual(g, g2)

    def test_exc(self):
        """ Opening dangling soft link results in KeyError """
        self.f['alias'] = SoftLink('new')
        with self.assertRaises(KeyError):
            self.f['alias']

class TestExternalLinks(TestCase):

    """
        Feature: Create and manage external links
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.ename = self.mktemp()
        self.ef = File(self.ename, 'w')
        self.ef.create_group('external')
        self.ef.close()

    def tearDown(self):
        if self.f:
            self.f.close()
        if self.ef:
            self.ef.close()

    def test_epath(self):
        """ External link paths attributes """
        el = ExternalLink('foo.hdf5', '/foo')
        self.assertEqual(el.filename, 'foo.hdf5')
        self.assertEqual(el.path, '/foo')

    def test_erepr(self):
        """ External link repr """
        el = ExternalLink('foo.hdf5','/foo')
        self.assertIsInstance(repr(el), basestring)

    def test_create(self):
        """ Creating external links """
        self.f['ext'] = ExternalLink(self.ename, '/external')
        grp = self.f['ext']
        self.ef = grp.file
        self.assertNotEqual(self.ef, self.f)
        self.assertEqual(grp.name, '/external')

    def test_exc(self):
        """ KeyError raised when attempting to open broken link """
        self.f['ext'] = ExternalLink(self.ename, '/missing')
        with self.assertRaises(KeyError):
            self.f['ext']

    # I would prefer IOError but there's no way to fix this as the exception
    # class is determined by HDF5.
    def test_exc_missingfile(self):
        """ KeyError raised when attempting to open missing file """
        self.f['ext'] = ExternalLink('mongoose.hdf5','/foo')
        with self.assertRaises(KeyError):
            self.f['ext']

    def test_close_file(self):
        """ Files opened by accessing external links can be closed 

        Issue 189.
        """
        self.f['ext'] = ExternalLink(self.ename, '/')
        grp = self.f['ext']
        f2 = grp.file
        f2.close()
        self.assertFalse(f2)

class TestExtLinkBugs(TestCase):

    """
        Bugs: Specific regressions for external links
    """

    def test_issue_212(self):
        """ Issue 212
    
        Fails with:

        AttributeError: 'SharedConfig' object has no attribute 'lapl'
        """
        def closer(x):
            def w():
                try:
                    if x:
                        x.close()
                except IOError:
                    pass
            return w
        orig_name = self.mktemp()
        new_name = self.mktemp()
        f = File(orig_name, 'w')
        self.addCleanup(closer(f))
        f.create_group('a')
        f.close()

        g = File(new_name, 'w')
        self.addCleanup(closer(g))
        g['link'] = ExternalLink(orig_name, '/')  # note root group
        g.close()

        h = File(new_name, 'r')
        self.addCleanup(closer(h))
        self.assertIsInstance(h['link']['a'], Group)


class TestCopy(TestCase):

    def setUp(self):
        self.f1 = File(self.mktemp(), 'w')
        self.f2 = File(self.mktemp(), 'w')
        
    def tearDown(self):
        if self.f1:
            self.f1.close()
        if self.f2:
            self.f2.close()
            
    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_path_to_path(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]

        self.f1.copy('foo', 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_path_to_group(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]
        baz = self.f1.create_group('baz')

        self.f1.copy('foo', baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1,2,3]))

        self.f1.copy('foo', self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_group_to_path(self):

        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]

        self.f1.copy(foo, 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1,2,3]))

        self.f2.copy(foo, 'foo')
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_group_to_group(self):

        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]
        baz = self.f1.create_group('baz')

        self.f1.copy(foo, baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1,2,3]))

        self.f1.copy(foo, self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_dataset(self):
        self.f1['foo'] = [1,2,3]
        foo = self.f1['foo']

        self.f1.copy(foo, 'bar')
        self.assertArrayEqual(self.f1['bar'], np.array([1,2,3]))

        self.f1.copy('foo', 'baz')
        self.assertArrayEqual(self.f1['baz'], np.array([1,2,3]))

        self.f1.copy('foo', self.f2)
        self.assertArrayEqual(self.f2['foo'], np.array([1,2,3]))

        self.f2.copy(self.f1['foo'], self.f2, 'bar')
        self.assertArrayEqual(self.f2['bar'], np.array([1,2,3]))
