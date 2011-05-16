import numpy as np
import sys

from .common import ut, TestCase
from h5py.highlevel import File, Group, SoftLink, HardLink, ExternalLink
from h5py.highlevel import Dataset, Datatype
from h5py import h5t

class BaseGroup(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCreate(TestCase):

    """
        Feature: New groups can be created via .create_group method
    """

    def test_create(self):
        """ Simple .create_group call """
        hfile = File(self.mktemp(), 'w')
        grp = hfile.create_group('foo')
        self.assertIsInstance(grp, Group)
        hfile.close()

    def test_create_intermediate(self):
        """ Intermediate groups can be created automatically """
        hfile = File(self.mktemp(), 'w')
        grp = hfile.create_group('foo/bar/baz')
        self.assertEqual(grp.name, '/foo/bar/baz')
        hfile.close()

    def test_create_exception(self):
        """ Name conflict causes group creation to fail with ValueError """
        hfile = File(self.mktemp(), 'w')
        hfile.create_group('foo')
        with self.assertRaises(ValueError):
            hfile.create_group('foo')
        hfile.close()

    def test_unicode(self):
        """ Unicode names are correctly stored """
        name = u"/Name\u4500"
        hfile = File(self.mktemp(), 'w')
        group = hfile.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_UTF8)
        hfile.close()

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

class TestRequire(TestCase):

    """
        Feature: Groups can be auto-created, or opened via .require_group
    """

    def test_open_existing(self):
        """ Existing group is opened and returned """
        hfile = File(self.mktemp(),'w')
        grp = hfile.create_group('foo')
        grp2 = hfile.require_group('foo')
        self.assertEqual(grp, grp2)
        hfile.close()

    def test_create(self):
        """ Group is created if it doesn't exist """
        hfile = File(self.mktemp(),'w')
        grp = hfile.require_group('foo')
        self.assertIsInstance(grp, Group)
        self.assertEqual(grp.name, '/foo')
        hfile.close()

    def test_require_exception(self):
        """ Opening conflicting object results in TypeError """
        hfile = File(self.mktemp(),'w')
        hfile.create_dataset('foo', (1,), 'f')
        with self.assertRaises(TypeError):
            hfile.require_group('foo')
        hfile.close()

class TestDelete(TestCase):

    """
        Feature: Objects can be unlinked via "del" operator
    """

    def test_delete(self):
        """ Object deletion via "del" """
        hfile = File(self.mktemp(),'w')
        self.addCleanup(hfile.close)
        
        hfile.create_group('foo')
        self.assertIn('foo', hfile)
        del hfile['foo']
        self.assertNotIn('foo', hfile)
        
    def test_nonexisting(self):
        """ Deleting non-existent object raises KeyError """
        hfile = File(self.mktemp(),'w')
        self.addCleanup(hfile.close)
        
        with self.assertRaises(KeyError):
            del hfile['foo']

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
        self.addCleanup(hfile.close)
        
        with self.assertRaises(KeyError):
            del hfile['foo']

class TestOpen(TestCase):

    """
        Feature: Objects can be opened via indexing syntax obj[name]
    """

    def test_open(self):
        """ Simple obj[name] opening """
        hfile = File(self.mktemp(),'w')
        grp = hfile.create_group('foo')
        grp2 = hfile['foo']
        grp3 = hfile['/foo']
        self.assertEqual(grp, grp2)
        self.assertEqual(grp, grp3)
        hfile.close()

    def test_nonexistent(self):
        """ Opening missing objects raises KeyError """
        hfile = File(self.mktemp(), 'w')
        with self.assertRaises(KeyError):
            hfile['foo']
        hfile.close()

    def test_reference(self):
        """ Objects can be opened by HDF5 object reference """
        hfile = File(self.mktemp(), 'w')
        grp = hfile.create_group('foo')
        grp2 = hfile[grp.ref]
        self.assertEqual(grp2, grp)
        hfile.close()

    # TODO: check that regionrefs also work with __getitem__

class TestRepr(TestCase):

    """
        Feature: Opened and closed groups provide a useful __repr__ string
    """

    def test_repr(self):
        """ Opened and closed groups provide a useful __repr__ string """
        hfile = File(self.mktemp(), 'w')
        g = hfile.create_group('foo')
        self.assertIsInstance(repr(g), basestring)
        g.id._close()
        self.assertIsInstance(repr(g), basestring)
        hfile.close()

class BaseMapping(TestCase):

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

    #TODO: See if this is really the right behavior
    @ut.expectedFailure
    def test_exc(self):
        """ "in" on closed group raises ValueError """
        self.f.close()
        with self.assertRaises(ValueError):
            'a' in self.f

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
        lst = [x for x in hfile]
        self.assertEqual(lst, [])

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

class TestGet(TestCase):

    """
        Feature: The .get method allows access to objects and metadata
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

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

class TestSoftLinks(TestCase):

    """
        Feature: Create and manage soft links with the high-level interface
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


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

    # TODO: This is raising KeyError; see if that's right
    @ut.expectedFailure
    def test_exc_missingfile(self):
        """ IOerror raised when attempting to open missing file """
        self.f['ext'] = ExternalLink('mongoose.hdf5','/foo')
        with self.assertRaises(IOError):
            self.f['ext']
































