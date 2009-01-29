#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-
from __future__ import with_statement

import unittest
import tempfile
import shutil
import os
import numpy

from h5py.highlevel import *
from h5py import *
from h5py.h5 import H5Error
from common import getfullpath, HDF5TestCase, api_18, api_16
import testfiles

class SliceFreezer(object):
    def __getitem__(self, args):
        return args

def skip(func):
    return None

HDFNAME = getfullpath("smpl_compound_chunked.hdf5")

TYPES1 = \
  [ "<i1", "<i2", "<i4", "<i8", ">i1", ">i2", ">i4", ">i8", "|i1", "|u1", 
    "<u1", "<u2", "<u4", "<u8", ">u1", ">u2", ">u4", ">u8",
    "<f4", "<f8", ">f4", ">f8", "<c8", "<c16", ">c8", ">c16"]

TYPES2 = ["|S1", "|S2", "|S33", "|V1", "|V2", "|V33"]


TYPES = TYPES1 + TYPES2

SHAPES = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]

 
class TestFile(HDF5TestCase):

    def setUp(self):
        newname = tempfile.mktemp('.hdf5')
        shutil.copy(HDFNAME, newname)
        self.fname = newname

    def tearDown(self):
        os.unlink(self.fname)

    def test_File_init_r(self):
        with File(self.fname, 'r') as f:
            self.assert_(isinstance(f["CompoundChunked"], Dataset))
            self.assertRaises(H5Error, f.create_group, "FooBar")
            self.assertEqual(f.mode, 'r')
            self.assertEqual(f.name, self.fname)

    def test_File_init_rp(self):
        with File(self.fname, 'r+') as f:
            self.assert_(isinstance(f["CompoundChunked"], Dataset))
            f.create_group("FooBar")
            self.assert_(isinstance(f["FooBar"], Group))
            self.assertEqual(f.mode, 'r+')
            self.assertEqual(f.name, self.fname)

    def test_File_init_a(self):
        with File(self.fname, 'a') as f:
            self.assert_(isinstance(f["CompoundChunked"], Dataset))
            f.create_group("FooBar")
            self.assert_(isinstance(f["FooBar"], Group))
            self.assertEqual(f.mode, 'a')
            self.assertEqual(f.name, self.fname)

    def test_File_init_w(self):
        with File(self.fname, 'w') as f:
            self.assert_("CompoundChunked" not in f)
            f.create_group("FooBar")
            self.assert_(isinstance(f["FooBar"], Group))
            self.assertEqual(f.mode, 'w')
            self.assertEqual(f.name, self.fname)

    def test_File_init_wm(self):
        self.assertRaises(H5Error, File, self.fname, 'w-')
        tmpname = tempfile.mktemp('.hdf5')
        f = File(tmpname,'w-')
        f.close()
        os.unlink(tmpname)

    def test_File_close(self):
        f = File(self.fname, 'r')
        self.assert_(f.id)
        f.close()
        self.assert_(not f.id)

    def test_File_flush(self):
        with File(self.fname) as f:
            f.flush()

    def test_File_special(self):
        f = File(self.fname, 'r')
        str(f)
        repr(f)
        f.close()
        str(f)
        repr(f)

    def test_AttributeManager(self):

        shapes = ((), (1,), (1,10), (10,1), (10,1,10), (9,9))
        attrs = [1, 2.0, 0, "Hello", " 129887A!!\t\t9(){; \t "]
        for shape in shapes:
            for dt in TYPES1:
                attrs += [numpy.arange(numpy.product(shape), dtype=dt).reshape(shape)]

        # Turn the list into a dictionary.  Names are str(indices).
        attrs = dict( [(str(idx), attr) for idx, attr in enumerate(attrs)] )

        with File(self.fname, 'w') as f:

            # Tests __setitem__ and __getitem__
            grp = f.create_group("Grp")
            for name, attr in attrs.iteritems():
                grp.attrs[name] = attr
                self.assert_(numpy.all(grp.attrs[name] == attr))

            # Test __str__ with attrs present
            str(grp.attrs)

            # Tests __iter__ for name comparison
            self.assertEqual(set(grp.attrs), set(attrs))

            # Tests iteritems()
            hattrs = dict(grp.attrs.iteritems())
            self.assertEqual(set(hattrs), set(attrs))  # check names
            for name in attrs:
                self.assert_(numpy.all(attrs[name] == hattrs[name]))  # check vals

            # Test __len__
            self.assertEqual(len(grp.attrs), len(attrs))

            # Tests __contains__ and __delitem__
            for name in list(grp.attrs):
                self.assert_(name in grp.attrs)
                del grp.attrs[name]
                self.assert_(not name in grp.attrs)

            self.assertEqual(len(grp.attrs), 0)

            # Test on closed object
            grp.id._close()
            str(grp.attrs)

class TestDataset(HDF5TestCase):

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)
  
    def test_special(self):
        """ Check object identity, hashing and string representation """
        dset1 = self.f.create_dataset('TEST', (10,10), '<i4')
        dset2 = self.f.create_dataset('TEST2', (10,10), '<i4')

        dset1_p = self.f['TEST']
        dset2_p = self.f['TEST2']

        self.assert_(dset1 != dset2)
        self.assert_(dset1 == dset1_p)

        self.assert_(hash(dset1) == hash(dset1_p))
        self.assert_(hash(dset1) != hash(dset2))

        repr(dset1)
        str(dset1)
        dset1.id._close()
        repr(dset1)
        str(dset1)
    
    def test_Dataset_order(self):
        """ Test order coercion """

        fortran = numpy.array([[1,2,3],[4,5,6]], order='F'),
        strided = numpy.arange(2*3*4, dtype=numpy.uint8)
        strided.shape=(2,3,4)
        strided.strides=(0,1,1)
        b = numpy.arange(2*3*4, dtype=numpy.uint8)
        view = numpy.ndarray(buffer=b, offset=2, shape=(2,4), dtype=numpy.uint8)

        for x in (fortran, strided, view):
            dset = self.f.create_dataset('TEST_DATA', data=x)
            self.assert_(numpy.all(dset[:] == x))
            del self.f['TEST_DATA']

    @api_18
    def test_Dataset_resize(self):
        """ Test extending datasets """

        self.output("")

        init_shapes = [(100,), (100,100), (150,100)]
        max_shapes = [(200,), (200,200), (None, 100)]
        chunks = [(10,), (10,10), (10,10)]

        final_shapes = {(100,): [ (150,), (200,) ],
                        (100,100): [(200,100), (200,200)],
                        (150,100): [ (200,100), (300,100), (500,100)] }

        illegal_shapes = {(100,): [(250,)], (100,100): [(250,100), (250,250)],
                          (150,100): [(200,150)] }

        for shape, maxshape, chunk in zip(init_shapes, max_shapes, chunks):
            srcarr = numpy.arange(numpy.product(shape)).reshape(shape)
            if "DS" in self.f:
                del self.f["DS"]
            ds = self.f.create_dataset("DS", data=srcarr, maxshape=maxshape, chunks=chunk)

            self.assertEqual(ds.shape, shape)

            for final_shape in final_shapes[shape]:
                self.output("    Extending %s to %s" % (shape, final_shape))
                newarr = numpy.arange(numpy.product(final_shape)).reshape(final_shape)
                ds.resize(final_shape)
                ds[...] = newarr
                self.assertEqual(ds.shape, final_shape)
                self.assert_(numpy.all(ds[...] == newarr))

            for illegal_shape in illegal_shapes[shape]:
                self.assertRaises(H5Error, ds.resize, illegal_shape)

    def test_Dataset_len_iter(self):
        """ Test new and old len(), iteration over rows """
        arr1 = numpy.arange(100).reshape((10,10))
        arr2 = numpy.ones(())

        d1 = self.f.create_dataset("D1", data=arr1)
        d2 = self.f.create_dataset("D2", data=arr2)
        d3 = self.f.create_dataset("D3", shape=(2**60, 2**50))

        self.assertEqual(len(arr1), len(d1))
        self.assertRaises(TypeError, d2, len)
        self.assertEqual(d3.len(), 2**60)

        for idx, (hval, nval) in enumerate(zip(d1, arr1)):
            self.assert_(numpy.all(hval == nval))
        
        self.assertEqual(idx+1, len(arr1))
        self.assertRaises(TypeError, list, d2)

    def test_slice_big(self):
        """ Test slices > 2**32 """
        self.output("")

        s = SliceFreezer()

        bases = [1024, 2**37, 2**60]
        shapes = [ (42,1), (100,100), (1,42), (1,1), (4,1025)]

        for base in bases:
            slices = [ s[base:base+x, base:base+y] for x, y in shapes]

            if "dset" in self.f:
                del self.f["dset"]

            dset = self.f.create_dataset("dset", (2**62, 2**62), '=f4', maxshape=(None,None))

            for shp, slc in zip(shapes, slices):
                self.output("    Testing base 2**%d" % numpy.log2(base))

                empty = numpy.zeros(shp)
                data = numpy.arange(numpy.product(shp), dtype='=f4').reshape(shp)

                dset[slc] = empty
                arr = dset[slc]
                self.assertEqual(arr.shape, shp)
                self.assert_(numpy.all(arr == empty), "%r \n\n %r" % (arr, empty))
                
                dset[slc] = data
                arr = dset[slc]
                self.assert_(numpy.all(arr == data), "%r \n\n %r" % (arr, data))


    @skip
    def test_slice_coords(self):
        """ Test slicing with CoordsList instances """

        space = (100,100)

        # These need to be increasing to make it easy to compare to the
        # NumPy reference array, which uses a boolean mask.
        selections = [0,1,15,101,102, 557, 664, 1024,9999]
        selections_list = [ selections, []]  # empty selection

        arr = numpy.arange(10000).reshape(space)
        
        dset = self.f.create_dataset('dset', data=arr)

        # Scalar selections
        for x in selections:
            sel = CoordsList(numpy.unravel_index(x,space))
            self.assertEqual(dset[sel], arr.flat[x])
            self.assert_(not isinstance(dset[sel], numpy.ndarray))

        for lst in selections_list:
            # Coordinate list selection
            sel = CoordsList([numpy.unravel_index(x,space) for x in lst])

            npy_sel = numpy.zeros(space, dtype='bool')
            for x in lst:
                npy_sel.flat[x] = True

            hresult = dset[sel]
            nresult = arr[npy_sel]
            self.assert_(numpy.all(hresult == nresult))
            self.assert_(isinstance(hresult, numpy.ndarray))
            self.assertEqual(hresult.dtype, nresult.dtype)
            self.assertEqual(hresult.shape, nresult.shape)

    def test_Dataset_exceptions(self):
        """ Test exceptions """
        # These trigger exceptions in H5Dread
        ref = numpy.ones((10,10), dtype='<i4')
        dsid = self.f.create_dataset('ds', data=ref)
        arr = numpy.ndarray((10,10), dtype='|S6') # incompatible datatype
        self.assertRaises(H5Error, dsid.id.read, h5s.ALL, h5s.ALL, arr)
        # or it'll segfault...


class TestGroup(HDF5TestCase):

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)

    def assert_equal_contents(self, a, b):
        self.assertEqual(set(a), set(b))

    def test_Group_init(self):
        
        grp = Group(self.f, "NewGroup", create=True)
        self.assert_("NewGroup" in self.f)
        grp2 = Group(self.f, "NewGroup")

        self.assertEqual(grp.name, "/NewGroup")

    def test_Group_create_group(self):

        grp = self.f.create_group("NewGroup")
        self.assert_("NewGroup" in self.f)
        self.assertRaises(H5Error, self.f.create_group, "NewGroup")

    def test_Group_create_dataset(self):

        ds = self.f.create_dataset("Dataset", shape=(10,10), dtype='<i4')
        self.assert_(isinstance(ds, Dataset))
        self.assert_("Dataset" in self.f)

    def test_Group_special(self):

        subgroups = ["Name1", " Name 1231987&*@&^*&#W  2  \t\t ", "name3",
                     "14", "!"]

        for name in subgroups:
            self.f.create_group(name)

        # __len__
        self.assertEqual(len(self.f), len(subgroups))
    
        # __contains__
        for name in subgroups:
            self.assert_(name in self.f)

        # __iter__
        self.assert_equal_contents(list(self.f), subgroups)

        # Dictionary compatibility methods
        self.assert_equal_contents(self.f.listnames(), subgroups)
        self.assert_equal_contents(list(self.f.iternames()), subgroups)

        self.assert_equal_contents(self.f.listobjects(), [self.f[x] for x in subgroups])
        self.assert_equal_contents(list(self.f.iterobjects()), [self.f[x] for x in subgroups])

        self.assert_equal_contents(self.f.listitems(), [(x, self.f[x]) for x in subgroups])
        self.assert_equal_contents(list(self.f.iteritems()), [(x, self.f[x]) for x in subgroups])

        # __delitem__
        for name in subgroups:
            self.assert_(name in self.f)
            del self.f[name]
            self.assert_(not name in self.f)

        self.assertEqual(len(self.f), 0)

        # __str__
        grp = self.f.create_group("Foobar")
        str(grp)
        repr(grp)
        grp.id._close()
        str(grp)
        repr(grp)

    def test_Group_setgetitem(self):
        # Also tests named types

        self.output('')
        for shape in SHAPES:
            for dt in TYPES1:

                self.output("    Assigning %s %s" % (dt, shape))

                # test arbitrary datasets
                dt_obj = numpy.dtype(dt)       
                arr = numpy.ones(shape, dtype=dt_obj)
                self.f["DS"] = arr
                harr = self.f["DS"]
                self.assert_(isinstance(harr, Dataset))
                self.assertEqual(harr.shape, shape)
                self.assertEqual(harr.dtype, dt_obj)
                self.assert_(numpy.all(harr.value == arr))

                # test named types
                self.f["TYPE"] = dt_obj
                htype = self.f["TYPE"]
                self.assert_(isinstance(htype, Datatype))
                self.assertEqual(htype.dtype, dt_obj)

                del self.f["DS"]
                del self.f["TYPE"]

        # Test creation of array from sequence
        seq = [1,-42,2,3,4,5,10]
        self.f["DS"] = seq
        harr = self.f["DS"]
        self.assert_(numpy.all(harr.value == numpy.array(seq)))
        del self.f["DS"]

        # test scalar -> 0-d dataset
        self.f["DS"] = 42
        harr = self.f["DS"]
        self.assert_(isinstance(harr, Dataset))
        self.assertEqual(harr.shape, ())
        self.assertEqual(harr.value, 42)
    
        # test hard linking
        self.f["DS1"] = self.f["DS"]
        info1 = h5g.get_objinfo(self.f.id,"DS")
        info2 = h5g.get_objinfo(self.f.id,"DS1")
        self.assertEqual(info1.fileno, info2.fileno)
        self.assertEqual(info1.objno, info2.objno)

        # test assignment of out-of-order arrays
        arr = numpy.array(numpy.arange(100).reshape((10,10)), order='F')
        self.f['FORTRAN'] = arr
        dset = self.f['FORTRAN']
        self.assert_(numpy.all(dset[:] == arr))
        self.assert_(dset[:].flags['C_CONTIGUOUS'])

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

        self.assertRaises(H5Error, self.f.require_group, 'bar')
        self.assertRaises(H5Error, self.f.require_dataset, 'foo', (10,10), '<i4')

        self.assertRaises(H5Error, self.f.require_dataset, 'bar', (10,11), '<i4')
        self.assertRaises(H5Error, self.f.require_dataset, 'bar', (10,10), '<c8')
        self.assertRaises(H5Error, self.f.require_dataset, 'bar', (10,10), '<i1', exact=True)

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
















