# -*- coding: utf-8 -*-
# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Group test module.

    Tests all methods and properties of Group objects, with the following
    exceptions:

    1. Method create_dataset is tested in module test_dataset
"""

import numpy as np
import os
import os.path
from collections.abc import MutableMapping
from tempfile import mkdtemp

import pytest

import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
from .common import ut, TestCase, make_name, is_main_thread

# If we can't encode unicode filenames, there's not much point failing tests
# which must fail
try:
    filename_encode(u"α")
except UnicodeEncodeError:
    NO_FS_UNICODE = True
else:
    NO_FS_UNICODE = False


class BaseGroup(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCreate(BaseGroup):

    """
        Feature: New groups can be created via .create_group method
    """

    def test_create_str(self):
        """ Simple .create_group call """
        grp = self.f.create_group(make_name())
        self.assertIsInstance(grp, Group)

    def test_create_bytes(self):
        grp = self.f.create_group(make_name().encode('utf8'))
        self.assertIsInstance(grp, Group)

    def test_create_intermediate_str(self):
        """ Intermediate groups can be created automatically """
        path = make_name("foo{}/bar/baz")
        grp = self.f.create_group(path)
        self.assertEqual(grp.name, "/" + path)

    def test_create_intermediate_bytes(self):
        path = make_name("foo{}/bar/baz")
        grp2 = self.f.create_group(path.encode("utf8"))
        self.assertEqual(grp2.name, "/" + path)

    def test_create_exception(self):
        """ Name conflict causes group creation to fail with ValueError """
        name = make_name()
        self.f.create_group(name)
        with self.assertRaises(ValueError):
            self.f.create_group(name)

    def test_unicode(self):
        """ Unicode names are correctly stored """
        n = "/" + make_name() + chr(0x4500)
        group = self.f.create_group(n)
        self.assertEqual(group.name, n)
        self.assertEqual(group.id.links.get_info(n.encode('utf8')).cset, h5t.CSET_UTF8)

    def test_unicode_default(self):
        """ Unicode names convertible to ASCII are stored as ASCII (issue 239)
        """
        n = make_name("/Hello, this is a name")
        group = self.f.create_group(n)
        self.assertEqual(group.name, n)
        self.assertEqual(group.id.links.get_info(n.encode('utf8')).cset, h5t.CSET_ASCII)

    def test_type(self):
        """ Names should be strings or bytes """
        with self.assertRaises(TypeError):
            self.f.create_group(1.)

    def test_appropriate_low_level_id(self):
        " Binding a group to a non-group identifier fails with ValueError "
        dset = self.f.create_dataset(make_name(), [1], "f4")
        with self.assertRaises(ValueError):
            Group(dset.id)

class TestDatasetAssignment(BaseGroup):

    """
        Feature: Datasets can be created by direct assignment of data
    """

    def test_ndarray(self):
        """ Dataset auto-creation by direct assignment """
        name = make_name()
        data = np.ones((4,4),dtype='f')
        self.f[name] = data
        self.assertIsInstance(self.f[name], Dataset)
        self.assertArrayEqual(self.f[name][...], data)

    def test_name_bytes(self):
        data = np.ones((4, 4), dtype='f')
        n = make_name().encode('utf8')
        self.f[n] = data
        self.assertIsInstance(self.f[n], Dataset)

class TestDtypeAssignment(BaseGroup):

    """
        Feature: Named types can be created by direct assignment of dtypes
    """

    def test_dtype(self):
        """ Named type creation """
        name = make_name()
        dtype = np.dtype('|S10')
        self.f[name] = dtype
        self.assertIsInstance(self.f[name], Datatype)
        self.assertEqual(self.f[name].dtype, dtype)

    def test_name_bytes(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        n = make_name().encode('utf8')
        self.f[n] = dtype
        self.assertIsInstance(self.f[n], Datatype)


class TestRequire(BaseGroup):

    """
        Feature: Groups can be auto-created, or opened via .require_group
    """

    def test_open_existing(self):
        """ Existing group is opened and returned """
        name = make_name()
        grp = self.f.create_group(name)
        grp2 = self.f.require_group(name)
        self.assertEqual(grp2, grp)

        grp3 = self.f.require_group(name.encode('utf8'))
        self.assertEqual(grp3, grp)

    def test_create(self):
        """ Group is created if it doesn't exist """
        name = make_name()
        grp = self.f.require_group(name)
        self.assertIsInstance(grp, Group)
        self.assertEqual(grp.name, '/' + name)

    def test_require_exception(self):
        """ Opening conflicting object results in TypeError """
        name = make_name()
        self.f.create_dataset(name, (1,), 'f')
        with self.assertRaises(TypeError):
            self.f.require_group(name)

    def test_intermediate_create_dataset(self):
        """ Intermediate is created if it doesn't exist """
        name = make_name("foo")
        dt = h5py.string_dtype()
        self.f.require_dataset(f"{name}/bar/baz", (1,), dtype=dt)
        group = self.f.get(name)
        assert isinstance(group, Group)
        group = self.f.get(f"{name}/bar")
        assert isinstance(group, Group)

    def test_intermediate_create_group(self):
        name = make_name("foo")
        self.f.require_group(f"{name}/bar/baz")
        group = self.f.get(name)
        assert isinstance(group, Group)
        group = self.f.get(f"{name}/bar")
        assert isinstance(group, Group)
        group = self.f.get(f"{name}/bar/baz")
        assert isinstance(group, Group)

    def test_require_shape(self):
        n = make_name("foo{}/resizable")
        ds = self.f.require_dataset(n, shape=(0, 3), maxshape=(None, 3), dtype=int)
        ds.resize(20, axis=0)
        self.f.require_dataset(n, shape=(0, 3), maxshape=(None, 3), dtype=int)
        self.f.require_dataset(n, shape=(20, 3), dtype=int)
        with self.assertRaises(TypeError):
            self.f.require_dataset(n, shape=(0, 0), maxshape=(3, None), dtype=int)
        with self.assertRaises(TypeError):
            self.f.require_dataset(n, shape=(0, 0), maxshape=(None, 5), dtype=int)
        with self.assertRaises(TypeError):
            self.f.require_dataset(n, shape=(0, 0), maxshape=(None, 5, 2), dtype=int)
        with self.assertRaises(TypeError):
            self.f.require_dataset(n, shape=(10, 3), dtype=int)


class TestDelete(BaseGroup):

    """
        Feature: Objects can be unlinked via "del" operator
    """

    def test_delete(self):
        """ Object deletion via "del" """
        name = make_name()
        self.f.create_group(name)
        self.assertIn(name, self.f)
        del self.f[name]
        self.assertNotIn(name, self.f)

    def test_nonexisting(self):
        """ Deleting non-existent object raises KeyError """
        with self.assertRaises(KeyError):
            del self.f['foo']

    def test_readonly_delete_exception(self):
        """ Deleting object in readonly file raises KeyError """
        # Note: it is impossible to restore the old behavior (ValueError)
        # without breaking the above test (non-existing objects)
        fname = self.mktemp()
        hfile = File(fname, 'w')
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
        name = make_name()
        grp = self.f.create_group(name)
        grp2 = self.f[name]
        grp3 = self.f[f"/{name}"]
        self.assertEqual(grp, grp2)
        self.assertEqual(grp, grp3)

    def test_nonexistent(self):
        """ Opening missing objects raises KeyError """
        with self.assertRaises(KeyError):
            self.f["notexist"]

    def test_reference(self):
        """ Objects can be opened by HDF5 object reference """
        grp = self.f.create_group(make_name())
        grp2 = self.f[grp.ref]
        self.assertEqual(grp2, grp)

    def test_reference_numpyobj(self):
        """ Object can be opened by numpy.object_ containing object ref

        Test for issue 181, issue 202.
        """
        g = self.f.create_group(make_name("g"))

        dt = np.dtype([('a', 'i'),('b', h5py.ref_dtype)])
        dset = self.f.create_dataset(make_name("x"), (1,), dt)

        dset[0] =(42,g.ref)
        data = dset[0]
        self.assertEqual(self.f[data[1]], g)

    def test_invalid_ref(self):
        """ Invalid region references should raise an exception """
        ref = h5py.h5r.Reference()

        with self.assertRaises(ValueError):
            self.f[ref]

    @pytest.mark.thread_unsafe(reason="FIXME #2672 does not raise")
    def test_deleted_ref(self):
        """ References to deleted objects should raise an exception """
        name = make_name()
        self.f.create_group(name)
        ref = self.f[name].ref
        del self.f[name]

        with self.assertRaises(Exception):
            self.f[ref]

    def test_path_type_validation(self):
        """ Access with non bytes or str types should raise an exception """
        self.f.create_group(make_name())

        with self.assertRaises(TypeError):
            self.f[0]

        with self.assertRaises(TypeError):
            self.f[...]

    # TODO: check that regionrefs also work with __getitem__

class TestRepr(BaseGroup):
    """Opened and closed groups provide a useful __repr__ string"""

    def test_repr(self):
        """ Opened and closed groups provide a useful __repr__ string """
        name = make_name()
        g = self.f.create_group(name)
        self.assertIsInstance(repr(g), str)
        g.id._close()
        self.assertIsInstance(repr(g), str)
        g = self.f[name]
        if is_main_thread():
            # Closing the file shouldn't break it
            self.f.close()
            self.assertIsInstance(repr(g), str)

class BaseMapping(BaseGroup):

    """
        Base class for mapping tests
    """
    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ('a', 'b', 'c', 'd')
        for x in self.groups:
            self.f.create_group(x)
        self.f['x'] = h5py.SoftLink('/mongoose')
        self.groups = self.groups + ('x',)

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

    def test_len_after_create_group(self):
        # Can't use a shared file when running in pytest-run-parallel
        with File(self.mktemp(), 'w') as f:
            self.assertEqual(len(f), 0)
            f.create_group("x")
            self.assertEqual(len(f), 1)
            f.create_group("y")
            self.assertEqual(len(f), 2)
            del f["x"]
            self.assertEqual(len(f), 1)

class TestContains(BaseGroup):

    """
        Feature: The Python "in" builtin tests for membership
    """

    def test_contains(self):
        """ "in" builtin works for membership (byte and Unicode) """
        name = make_name()
        self.f.create_group(name)
        self.assertIn(name.encode("utf-8"), self.f)
        self.assertIn(name, self.f)
        self.assertIn(f"/{name}".encode("utf-8"), self.f)
        self.assertIn(f"/{name}", self.f)
        self.assertNotIn(b'mongoose', self.f)
        self.assertNotIn('mongoose', self.f)

    def test_closed(self):
        """ "in" on closed File returns False (see also issue 174) """
        f = File(self.mktemp(), 'w')
        f.create_group('a')
        self.assertTrue(b'a' in f)
        self.assertTrue('a' in f)
        f.close()
        self.assertFalse(b'a' in f)
        self.assertFalse('a' in f)

    def test_empty(self):
        """ Empty strings work properly and aren't contained """
        self.assertNotIn('', self.f)
        self.assertNotIn(b'', self.f)

    def test_dot(self):
        """ Current group "." is always contained """
        self.assertIn(b'.', self.f)
        self.assertIn('.', self.f)

    def test_root(self):
        """ Root group (by itself) is contained """
        self.assertIn(b'/', self.f)
        self.assertIn('/', self.f)

    def test_trailing_slash(self):
        """ Trailing slashes are unconditionally ignored """
        g = make_name("g")
        d = make_name("d")
        self.f.create_group(g)
        self.f[d] = 42
        self.assertIn(f"/{g}/", self.f)
        self.assertIn(f"{g}/", self.f)
        self.assertIn(f"/{d}/", self.f)
        self.assertIn(f"{d}/", self.f)

    def test_softlinks(self):
        """ Broken softlinks are contained, but their members are not """
        name = make_name()
        self.f.create_group(name)
        self.f[f'/{name}/soft'] = h5py.SoftLink('/mongoose')
        self.f[f'/{name}/external'] = h5py.ExternalLink('mongoose.hdf5', '/mongoose')
        self.assertIn(f"/{name}/soft", self.f)
        self.assertNotIn(f"/{name}/soft/something", self.f)
        self.assertIn(f"/{name}/external", self.f)
        self.assertNotIn(f"/{name}/external/something", self.f)

    def test_oddball_paths(self):
        """ Technically legitimate (but odd-looking) paths """
        x = make_name('x')
        dset = make_name("dset")

        self.f.create_group(f"{x}/y/z")
        self.f[dset] = 42
        self.assertIn('/', self.f)
        self.assertIn('//', self.f)
        self.assertIn('///', self.f)
        self.assertIn('.///', self.f)
        self.assertIn('././/', self.f)
        grp = self.f[x]
        self.assertIn(f'.//{x}/y/z', self.f)
        self.assertNotIn(f'.//{x}/y/z', grp)
        self.assertIn(f'{x}///', self.f)
        self.assertIn(f'./{x}///', self.f)
        self.assertIn(f'{dset}///', self.f)
        self.assertIn(f'/{dset}//', self.f)

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

class TestTrackOrder(BaseGroup):
    def populate(self, g):
        for i in range(100):
            # Mix group and dataset creation.
            if i % 10 == 0:
                g.create_group(str(i))
            else:
                g[str(i)] = [i]

    def test_track_order(self):
        g = self.f.create_group(make_name(), track_order=True)  # creation order
        self.populate(g)

        ref = [str(i) for i in range(100)]
        self.assertEqual(list(g), ref)
        self.assertEqual(list(reversed(g)), list(reversed(ref)))

    def test_no_track_order(self):
        g = self.f.create_group(make_name(), track_order=False)  # name alphanumeric
        self.populate(g)

        ref = sorted([str(i) for i in range(100)])
        self.assertEqual(list(g), ref)
        self.assertEqual(list(reversed(g)), list(reversed(ref)))

class TestPy3Dict(BaseMapping):

    def test_keys(self):
        """ .keys provides a key view """
        kv = getattr(self.f, 'keys')()
        ref = self.groups
        self.assertSameElements(list(kv), ref)
        self.assertSameElements(list(reversed(kv)), list(reversed(ref)))

        for x in self.groups:
            self.assertIn(x, kv)
        self.assertEqual(len(kv), len(self.groups))

    def test_values(self):
        """ .values provides a value view """
        vv = getattr(self.f, 'values')()
        ref = [self.f.get(x) for x in self.groups]
        self.assertSameElements(list(vv), ref)
        self.assertSameElements(list(reversed(vv)), list(reversed(ref)))

        self.assertEqual(len(vv), len(self.groups))
        for x in self.groups:
            self.assertIn(self.f.get(x), vv)

    def test_items(self):
        """ .items provides an item view """
        iv = getattr(self.f, 'items')()
        ref = [(x,self.f.get(x)) for x in self.groups]
        self.assertSameElements(list(iv), ref)
        self.assertSameElements(list(reversed(iv)), list(reversed(ref)))

        self.assertEqual(len(iv), len(self.groups))
        for x in self.groups:
            self.assertIn((x, self.f.get(x)), iv)

class TestAdditionalMappingFuncs(BaseMapping):
    """
    Feature: Other dict methods (pop, pop_item, clear, update, setdefault) are
    available.
    """
    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        for x in ('/test/a', '/test/b', '/test/c', '/test/d'):
            self.f.create_group(x)
        self.group = self.f['test']

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_pop_item(self):
        """.pop_item removes an item"""
        g = self.f.create_group(make_name())
        g.create_group("a")
        g.create_group("b")
        k, _ = g.popitem()
        assert k in ("a", "b")
        self.assertNotIn(k, g)

        k2, _ = g.popitem()
        assert k2 == "b" if k == "a" else "a"
        self.assertNotIn(k2, g)

        # popitem() from an empty group raises
        with self.assertRaises(KeyError):
            g.popitem()

    def test_pop(self):
        """.pop returns and removes specified item"""
        g = self.f.create_group(make_name())
        g.create_group("a")
        g.create_group("b")
        g.pop("a")
        self.assertNotIn("a", g)
        self.assertIn("b", g)

    def test_pop_default(self):
        """.pop falls back to default"""
        # e shouldn't exist as a group
        value = self.group.pop('e', None)
        self.assertEqual(value, None)

    def test_pop_raises(self):
        """.pop raises KeyError for non-existence"""
        # e shouldn't exist as a group
        with self.assertRaises(KeyError):
            key = self.group.pop('e')

    def test_clear(self):
        """.clear removes groups"""
        g = self.f.create_group(make_name())
        g.create_group('a')
        g.create_group('b')
        self.assertEqual(len(g), 2)
        g.clear()
        self.assertEqual(len(g), 0)

    def test_update_dict(self):
        """.update works with dict"""
        g = self.f.create_group(make_name())
        new_items = {'e': np.array([42])}
        g.update(new_items)
        self.assertIn('e', g)

    def test_update_iter(self):
        """.update works with list"""
        g = self.f.create_group(make_name())
        new_items = [
            ('e', np.array([42])),
            ('f', np.array([42]))
        ]
        g.update(new_items)
        self.assertIn('e', g)

    def test_update_kwargs(self):
        """.update works with kwargs"""
        g = self.f.create_group(make_name())
        new_items = {'e': np.array([42])}
        g.update(**new_items)
        self.assertIn('e', g)

    def test_setdefault(self):
        """.setdefault gets group if it exists"""
        value = self.group.setdefault('a')
        self.assertEqual(value, self.group.get('a'))

    def test_setdefault_with_default(self):
        """.setdefault gets default if group doesn't exist"""
        # 42 used as groups should be strings
        value = self.group.setdefault(make_name(), np.array([42]))
        self.assertEqual(value, 42)

    def test_setdefault_no_default(self):
        """
        .setdefault gets None if group doesn't exist, but as None isn't defined
        as data for a dataset, this should raise a TypeError.
        """
        # e shouldn't exist as a group
        with self.assertRaises(TypeError):
            self.group.setdefault('e')


class TestGet(BaseGroup):

    """
        Feature: The .get method allows access to objects and metadata
    """

    def test_get_default(self):
        """ Object is returned, or default if it doesn't exist """
        name = make_name()
        default = object()
        out = self.f.get('mongoose', default)
        self.assertIs(out, default)

        grp = self.f.create_group(name)
        out = self.f.get(name.encode('utf8'))
        self.assertEqual(out, grp)

    def test_get_class(self):
        """ Object class is returned with getclass option """
        foo = make_name("foo")
        bar = make_name("bar")
        baz = make_name("baz")

        self.f.create_group(foo)
        out = self.f.get(foo, getclass=True)
        self.assertEqual(out, Group)

        self.f.create_dataset(bar, (4,), "f4")
        out = self.f.get(bar, getclass=True)
        self.assertEqual(out, Dataset)

        self.f[baz] = np.dtype('|S10')
        out = self.f.get(baz, getclass=True)
        self.assertEqual(out, Datatype)

    def test_get_link_class(self):
        """ Get link classes """
        hard = make_name("hard")
        soft = make_name("soft")
        external = make_name("external")

        default = object()
        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group(hard)
        self.f[soft] = sl
        self.f[external] = el

        out_hl = self.f.get(hard, default, getlink=True, getclass=True)
        out_sl = self.f.get(soft, default, getlink=True, getclass=True)
        out_el = self.f.get(external, default, getlink=True, getclass=True)

        self.assertEqual(out_hl, HardLink)
        self.assertEqual(out_sl, SoftLink)
        self.assertEqual(out_el, ExternalLink)

    def test_get_link(self):
        """ Get link values """
        hard = make_name("hard")
        soft = make_name("soft")
        external = make_name("external")

        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group(hard)
        self.f[soft] = sl
        self.f[external] = el

        out_hl = self.f.get(hard, getlink=True)
        out_sl = self.f.get(soft, getlink=True)
        out_el = self.f.get(external, getlink=True)

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
        # do not make assumption on iteration order
        l = []
        x = self.f.visit(lambda x: l.append(x) or -1)
        assert x == -1 and len(l) == 1 and l[0] in self.groups

        l = []
        comp = [(x, self.f[x]) for x in self.groups]
        x = self.f.visititems(lambda x, y: l.append((x,y)) or -1)
        assert x == -1 and len(l) == 1 and l[0] in comp

class TestVisitLinks(TestCase):
    """
        Feature: The .visit_links and .visititems_links methods allow iterative access to
        links contained in the group and its subgroups.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = [
            'grp1', 'grp1/grp11', 'grp1/grp12', 'grp2', 'grp2/grp21', 'grp2/grp21/grp211'
            ]
        self.links = [
            'linkto_grp1', 'grp1/linkto_grp11', 'grp1/linkto_grp12', 'linkto_grp2', 'grp2/linkto_grp21', 'grp2/grp21/linkto_grp211'
        ]
        for g, l in zip(self.groups, self.links, strict=True):
            self.f.create_group(g)
            self.f[l] = SoftLink(f'/{g}')

    def tearDown(self):
        self.f.close()

    def test_visit_links(self):
        """ All subgroups and links are visited """
        l = []
        self.f.visit_links(l.append)
        self.assertSameElements(l, self.groups + self.links)

    def test_visititems(self):
        """ All links are visited """
        l = []
        comp = [(x, type(self.f.get(x, getlink=True))) for x in self.groups + self.links]
        self.f.visititems_links(lambda x, y: l.append((x, type(y))))
        self.assertSameElements(comp, l)

    def test_bailout(self):
        """ Returning a non-None value immediately aborts iteration """
        # do not make assumption on iteration order
        l = []
        x = self.f.visit_links(lambda x: l.append(x) or -1)
        assert x == -1 and len(l) == 1

        l = []
        x = self.f.visititems_links(lambda x, y: l.append((x,y)) or -1)
        assert x == -1 and len(l) == 1

class Visitor:
    """ Class for exercise 'visit' and 'visititems' methods """

    def __init__(self):
        self._names = []

    def __call__(self, name, obj=None):
        self._names.append(name)

    @property
    def names(self):
        return self._names

class TestLexicographic(TestCase):
    """ Test ascending lexicographic order traversal of the 'visit*' methods.

        This semantics is set by the following default args in
        h5py.h5o.visit(..., idx_type=H5_INDEX_NAME, order=H5_ITER_INC, ...)
        h5py.h5l.visit(..., idx_type=H5_INDEX_NAME, order=H5_ITER_INC, ...)
    """

    import operator
    split_parts = operator.methodcaller('split', '/')

    def setUp(self):
        """ Populate example hdf5 file, with track_order=True """

        self.f = File(self.mktemp(), 'w-', track_order=True)
        self.f.create_dataset('b', (10,), "f4")

        grp = self.f.create_group('B', track_order=True)
        grp.create_dataset('b', (10,), "f4")
        grp.create_dataset('a', (10,), "f4")

        grp = self.f.create_group('z', track_order=True)
        grp.create_dataset('b', (10,), "f4")
        grp.create_dataset('a', (10,), "f4")

        self.f.create_dataset('a', (10,), "f4")
        # note that 'z-' < 'z/...' but traversal order is ['z', 'z/...', 'z-']
        self.f.create_dataset('z-', (10,), "f4")

        # create some links
        self.f['A/x'] = self.f['B/b']
        self.f['y'] = self.f['z/a']
        self.f['A$'] = self.f['y']
        self.f['A/B/C'] = self.f['A']
        self.f['A/a'] = self.f['A']

    def test_nontrivial_sort_visit(self):
        """check that test example is not trivially sorted"""
        v = Visitor()
        self.f.visit(v)
        assert v.names != sorted(v.names)

    def test_visit(self):
        """check that File.visit iterates in lexicographic order"""
        v = Visitor()
        self.f.visit(v)
        assert v.names == sorted(v.names, key=self.split_parts)

    def test_visit_links(self):
        """check that File.visit_links iterates in lexicographic order"""
        v = Visitor()
        self.f.visit_links(v)
        assert v.names == sorted(v.names, key=self.split_parts)

    def test_visititems(self):
        """check that File.visititems iterates in lexicographic order"""
        v = Visitor()
        self.f.visititems(v)
        assert v.names == sorted(v.names, key=self.split_parts)

    def test_visititems_links(self):
        """check that File.visititems_links iterates in lexicographic order"""
        v = Visitor()
        self.f.visititems_links(v)
        assert v.names == sorted(v.names, key=self.split_parts)

    def test_visit_group(self):
        """check that Group.visit iterates in lexicographic order"""
        v = Visitor()
        self.f['A'].visit(v)
        assert v.names == sorted(v.names, key=self.split_parts)

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
        self.assertIsInstance(repr(sl), str)

    def test_create(self):
        """ Create new soft link by assignment """
        new = make_name("new")
        alias = make_name("alias")

        g = self.f.create_group(new)
        sl = SoftLink(f"/{new}")
        self.f[alias] = sl
        g2 = self.f[alias]
        self.assertEqual(g, g2)

    def test_exc(self):
        """ Opening dangling soft link results in KeyError """
        name = make_name()
        self.f[name] = SoftLink('new')
        with self.assertRaises(KeyError):
            self.f[name]


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
        self.assertIsInstance(repr(el), str)

    def test_create(self):
        """ Creating external links """
        name = make_name()
        self.f[name] = ExternalLink(self.ename, '/external')
        grp = self.f[name]
        self.ef = grp.file
        self.assertNotEqual(self.ef, self.f)
        self.assertEqual(grp.name, '/external')

    def test_exc(self):
        """ KeyError raised when attempting to open broken link """
        name = make_name()
        self.f[name] = ExternalLink(self.ename, '/missing')
        with self.assertRaises(KeyError):
            self.f[name]

    # I would prefer OSError but there's no way to fix this as the exception
    # class is determined by HDF5.
    def test_exc_missingfile(self):
        """ KeyError raised when attempting to open missing file """
        name = make_name()
        self.f[name] = ExternalLink('mongoose.hdf5','/foo')
        with self.assertRaises(KeyError):
            self.f[name]

    def test_close_file(self):
        """ Files opened by accessing external links can be closed

        Issue 189.
        """
        name = make_name()
        self.f[name] = ExternalLink(self.ename, '/')
        grp = self.f[name]
        f2 = grp.file
        f2.close()
        self.assertFalse(f2)

    @ut.skipIf(NO_FS_UNICODE, "No unicode filename support")
    def test_unicode_encode(self):
        """
        Check that external links encode unicode filenames properly
        Testing issue #732
        """
        ext_filename = os.path.join(mkdtemp(), u"α.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('external')
        self.f[make_name()] = ExternalLink(ext_filename, '/external')

    @ut.skipIf(NO_FS_UNICODE, "No unicode filename support")
    def test_unicode_decode(self):
        """
        Check that external links decode unicode filenames properly
        Testing issue #732
        """
        name = make_name()
        ext_filename = os.path.join(mkdtemp(), u"α.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('external')
            ext_file["external"].attrs["ext_attr"] = "test"
        self.f[name] = ExternalLink(ext_filename, '/external')
        self.assertEqual(self.f[name].attrs["ext_attr"], "test")

    def test_unicode_hdf5_path(self):
        """
        Check that external links handle unicode hdf5 paths properly
        Testing issue #333
        """
        name = make_name()
        ext_filename = os.path.join(mkdtemp(), "external.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('α')
            ext_file["α"].attrs["ext_attr"] = "test"
        self.f[name] = ExternalLink(ext_filename, '/α')
        self.assertEqual(self.f[name].attrs["ext_attr"], "test")


def test_get_elink_mode_arg(tmp_path):
    """Check that external link is opened with modified access mode"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w") as external_file:
        external_file.create_group("/group")

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w") as main_file:
        main_file["/external"] = h5py.ExternalLink(str(external_filepath), "/group")

    with File(main_filepath, "a") as main_file:
        ext_group = main_file.get("/external", elink_mode="r")
        assert ext_group.file.mode == "r"
        with pytest.raises(ValueError):
            ext_group['data'] = 1  # Writing fails


def test_get_elink_mode_arg_exception(tmp_path):
    """Check that external links open in write mode from read-only file fails"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w") as external_file:
        external_file.create_group("/group")

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w") as main_file:
        main_file["/external"] = h5py.ExternalLink(str(external_filepath), "/group")

    with File(main_filepath, "r") as main_file:
        with pytest.raises(ValueError):
            main_file.get("external", elink_mode="r+")


@pytest.mark.parametrize("file_swmr", (False, True))
@pytest.mark.parametrize("elink_swmr", (False, True))
def test_get_elink_swmr_arg_from_ro_file(tmp_path, file_swmr, elink_swmr):
    """Check that external links are opened with custom read-only SWMR mode"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w", libver="latest") as external_file:
        external_file.create_group("/group")

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w", libver="latest") as main_file:
        main_file["/external"] = h5py.ExternalLink(str(external_filepath), "/group")

    with File(main_filepath, "r", swmr=file_swmr) as main_file:
        ext_group = main_file.get("external", elink_swmr=elink_swmr)
        assert ext_group.file.mode == "r"
        assert ext_group.file.swmr_mode == elink_swmr


@pytest.mark.parametrize("elink_swmr", (False, True))
def test_get_elink_swmr_arg_from_rw_file(tmp_path, elink_swmr):
    """Check that external links are opened with custom read-only SWMR mode"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w", libver="latest") as external_file:
        external_file.create_group("/group")

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w", libver="latest") as main_file:
        main_file["/external"] = h5py.ExternalLink(str(external_filepath), "/group")

    with File(main_filepath, "a") as main_file:
        ext_group = main_file.get("external", elink_mode="r", elink_swmr=elink_swmr)
        assert ext_group.file.mode == "r"
        assert ext_group.file.swmr_mode == elink_swmr


def test_get_elink_swmr_arg_from_rw_swmr_file(tmp_path):
    """Check that exceptions are raised when changing access mode from a file opened in read-write SWMR mode"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w", libver="latest") as external_file:
        external_file.create_group("/group")

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w", libver="latest") as main_file:
        main_file["/external"] = h5py.ExternalLink(str(external_filepath), "/group")
        main_file.swmr_mode = True

        with pytest.raises(ValueError):
            main_file.get("external", elink_mode="r")

        with pytest.raises(ValueError):
            main_file.get("external", elink_swmr=False)


@pytest.mark.parametrize("file_locking", (False, True, "best-effort"))
@pytest.mark.parametrize(
    "elink_locking,file_locking_props",
    [
        (False, (0, 0)),
        (True, (1, 0)),
        ("best-effort", (1, 1)),
    ]
)
def test_get_elink_locking_arg(tmp_path, file_locking, elink_locking, file_locking_props):
    """Check that external links are opened with custom file locking"""
    external_filepath = tmp_path / make_name("external{}.h5")
    with File(external_filepath, "w") as external_file:
        external_file.create_group("/group")
        external_file["/group/data"] = 1, 2, 3

    main_filepath = tmp_path / make_name("main{}.h5")
    with File(main_filepath, "w") as main_file:
        main_file["external"] = ExternalLink(str(external_filepath), "/group")

    with File(main_filepath, "r", locking=file_locking) as main_file:
        cls = main_file.get("/external/data", getclass=True, elink_locking=elink_locking)
        assert cls is Dataset

        link = main_file.get("/external/data", getlink=True, elink_locking=elink_locking)
        assert isinstance(link, HardLink)

        ext_group = main_file.get("external", elink_locking=elink_locking)
        access_plist = ext_group.file.id.get_access_plist()
        assert access_plist.get_file_locking() == file_locking_props


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
                except OSError:
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

    def test_copy_path_to_path(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        foo1 = self.f1.create_group(name1)
        foo1['bar'] = [1,2,3]

        self.f1.copy(name1, name2)
        foo2 = self.f1[name2]
        self.assertIsInstance(foo2, Group)
        self.assertArrayEqual(foo2['bar'], np.array([1,2,3]))

    def test_copy_path_to_group(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        foo1 = self.f1.create_group(name1)
        foo1['bar'] = [1,2,3]
        baz = self.f1.create_group(name2)

        self.f1.copy(name1, baz)
        foo2 = self.f1[name2]
        self.assertIsInstance(foo2, Group)
        self.assertArrayEqual(foo2[f"{name1}/bar"], np.array([1,2,3]))

        self.f1.copy(name1, self.f2['/'])
        self.assertIsInstance(self.f2[name1], Group)
        self.assertArrayEqual(self.f2[f"{name1}/bar"], np.array([1,2,3]))

    def test_copy_group_to_path(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        foo1 = self.f1.create_group(name1)
        foo1['bar'] = [1,2,3]

        self.f1.copy(foo1, name2)
        foo2 = self.f1[name2]
        self.assertIsInstance(foo2, Group)
        self.assertArrayEqual(foo2['bar'], np.array([1,2,3]))

        self.f2.copy(foo1, name1)
        self.assertIsInstance(self.f2[name1], Group)
        self.assertArrayEqual(self.f2[f"{name1}/bar"], np.array([1,2,3]))

    def test_copy_group_to_group(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        foo1 = self.f1.create_group(name1)
        foo1['bar'] = [1,2,3]
        foo2 = self.f1.create_group(name2)

        self.f1.copy(foo1, foo2)
        foo2 = self.f1[name2]
        self.assertIsInstance(foo2, Group)
        self.assertArrayEqual(foo2[f"{name1}/bar"], np.array([1,2,3]))

        self.f1.copy(foo1, self.f2['/'])
        self.assertIsInstance(self.f2[f"/{name1}"], Group)
        self.assertArrayEqual(self.f2[f"{name1}/bar"], np.array([1,2,3]))

    def test_copy_dataset(self):
        name1 = make_name("foo")
        name2 = make_name("bar")
        name3 = make_name("baz")
        name4 = make_name("grp")

        self.f1[name1] = [1,2,3]
        foo = self.f1[name1]
        grp = self.f1.create_group(name4)

        self.f1.copy(foo, name2)
        self.assertArrayEqual(self.f1[name2], np.array([1,2,3]))

        self.f1.copy(name1, name3)
        self.assertArrayEqual(self.f1[name3], np.array([1,2,3]))

        self.f1.copy(foo, grp)
        self.assertArrayEqual(
            self.f1[f"/{name4}/{name1}"], np.array([1,2,3])
        )

        self.f1.copy(name1, self.f2)
        self.assertArrayEqual(self.f2[name1], np.array([1,2,3]))

        self.f2.copy(self.f1[name1], self.f2, name2)
        self.assertArrayEqual(self.f2[name2], np.array([1,2,3]))

    def test_copy_shallow(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        foo1 = self.f1.create_group(name1)
        bar = foo1.create_group('bar')
        foo1['qux'] = [1,2,3]
        bar['quux'] = [4,5,6]

        self.f1.copy(foo1, name2, shallow=True)
        baz = self.f1[name2]
        self.assertIsInstance(baz, Group)
        self.assertIsInstance(baz['bar'], Group)
        self.assertEqual(len(baz['bar']), 0)
        self.assertArrayEqual(baz['qux'], np.array([1,2,3]))

        self.f2.copy(foo1, name1, shallow=True)
        self.assertIsInstance(self.f2[f"/{name1}"], Group)
        self.assertIsInstance(self.f2[f"{name1}/bar"], Group)
        self.assertEqual(len(self.f2[f"{name1}/bar"]), 0)
        self.assertArrayEqual(self.f2[f"{name1}/qux"], np.array([1,2,3]))

    def test_copy_without_attributes(self):
        name1 = make_name("foo1")
        name2 = make_name("foo2")

        self.f1[name1] = [1,2,3]
        foo1 = self.f1[name1]
        foo1.attrs['bar'] = [4,5,6]

        self.f1.copy(foo1, name2, without_attrs=True)
        self.assertArrayEqual(self.f1[name2], np.array([1,2,3]))
        assert 'bar' not in self.f1[name2].attrs

        self.f2.copy(foo1, name2, without_attrs=True)
        self.assertArrayEqual(self.f2[name2], np.array([1,2,3]))
        assert 'bar' not in self.f2[name2].attrs

    def test_copy_soft_links(self):
        name1 = make_name("foo")
        name2 = make_name("bar")
        name3 = make_name("baz")

        self.f1[name2] = [1, 2, 3]
        foo = self.f1.create_group(name1)
        foo['qux'] = SoftLink(f"/{name2}")

        self.f1.copy(foo, name3, expand_soft=True)
        self.f2.copy(foo, name1, expand_soft=True)
        del self.f1[name2]

        self.assertIsInstance(self.f1[name3], Group)
        self.assertArrayEqual(self.f1[f"{name3}/qux"], np.array([1, 2, 3]))

        self.assertIsInstance(self.f2[f"/{name1}"], Group)
        self.assertArrayEqual(self.f2[f"{name1}/qux"], np.array([1, 2, 3]))

    def test_copy_external_links(self):
        name1 = make_name("foo")
        name2 = make_name("bar")
        name3 = make_name("baz")

        filename = self.mktemp()
        with File(filename, 'w') as f1:
            f1[name1] = [1,2,3]
            self.f2[name2] = ExternalLink(f1.filename, name1)

        self.assertArrayEqual(self.f2[name2], np.array([1,2,3]))

        self.f2.copy(name2, name3, expand_external=True)
        os.unlink(filename)
        self.assertArrayEqual(self.f2[name3], np.array([1,2,3]))

    def test_copy_refs(self):
        name1 = make_name("foo")
        name2 = make_name("bar")
        name3 = make_name("baz")
        name4 = make_name("qux")

        self.f1[name1] = [1,2,3]
        self.f1[name2] = [4,5,6]
        foo = self.f1[name1]
        bar = self.f1[name2]
        foo.attrs['bar'] = bar.ref

        self.f1.copy(foo, name3, expand_refs=True)
        self.assertArrayEqual(self.f1[name3], np.array([1,2,3]))
        baz_bar = self.f1[name3].attrs['bar']
        self.assertArrayEqual(self.f1[baz_bar], np.array([4,5,6]))
        # The reference points to a copy of bar, not to bar itself.
        self.assertNotEqual(self.f1[baz_bar].name, bar.name)

        self.f1.copy(name1, self.f2, name3, expand_refs=True)
        self.assertArrayEqual(self.f2[name3], np.array([1,2,3]))
        baz_bar = self.f2[name3].attrs['bar']
        self.assertArrayEqual(self.f2[baz_bar], np.array([4,5,6]))

        self.f1.copy('/', self.f2, name4, expand_refs=True)
        self.assertArrayEqual(self.f2[f"{name4}/{name1}"], np.array([1,2,3]))
        self.assertArrayEqual(self.f2[f"{name4}/{name2}"], np.array([4,5,6]))
        foo_bar = self.f2[f"{name4}/{name1}"].attrs['bar']
        self.assertArrayEqual(self.f2[foo_bar], np.array([4,5,6]))
        # There's only one copy of bar, which the reference points to.
        self.assertEqual(self.f2[foo_bar], self.f2[f"{name4}/{name2}"])


class TestMove(BaseGroup):

    """
        Feature: Group.move moves links in a file
    """

    def test_move_hardlink(self):
        """ Moving an object """
        x = make_name("x")
        y = make_name("y")
        z = make_name("z{}/nested/path")

        grp = self.f.create_group(x)
        self.f.move(x, y)
        self.assertEqual(self.f[y], grp)
        self.f.move(y, z)
        self.assertEqual(self.f[z], grp)

    def test_move_softlink(self):
        """ Moving a soft link """
        name1 = make_name("soft")
        name2 = make_name("new_soft")
        name3 = make_name("relative{}/path")

        self.f[name1] = h5py.SoftLink(name3)
        self.f.move(name1, name2)
        lnk = self.f.get(name2, getlink=True)
        self.assertEqual(lnk.path, name3)

    def test_move_conflict(self):
        """ Move conflict raises ValueError """
        x = make_name("x")
        y = make_name("y")

        self.f.create_group(x)
        self.f.create_group(y)
        with self.assertRaises(ValueError):
            self.f.move(x, y)

    def test_short_circuit(self):
        ''' Test that a null-move works '''
        name = make_name()
        self.f.create_group(name)
        self.f.move(name, name)


class TestMutableMapping(BaseGroup):
    '''Tests if the registration of Group as a MutableMapping
    behaves as expected
    '''
    def test_resolution(self):
        assert issubclass(Group, MutableMapping)
        grp = self.f.create_group(make_name())
        assert isinstance(grp, MutableMapping)

    def test_validity(self):
        '''
        Test that the required functions are implemented.
        '''
        Group.__getitem__
        Group.__setitem__
        Group.__delitem__
        Group.__iter__
        Group.__len__
