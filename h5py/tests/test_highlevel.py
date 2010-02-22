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
import numpy as np

import os.path as op

import h5py
from h5py.highlevel import *
from h5py import *
from common import TestCasePlus, api_18, api_16, res, TestCasePlus
import common
import testfiles

class SliceFreezer(object):
    def __getitem__(self, args):
        return args

TYPES1 = \
  [ "<i1", "<i2", "<i4", "<i8", ">i1", ">i2", ">i4", ">i8", "|i1", "|u1", 
    "<u1", "<u2", "<u4", "<u8", ">u1", ">u2", ">u4", ">u8",
    "<f4", "<f8", ">f4", ">f8", "<c8", "<c16", ">c8", ">c16"]

TYPES2 = ["|S1", "|S2", "|S33", "|V1", "|V2", "|V33"]


TYPES = TYPES1 + TYPES2

SHAPES = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]


def require_unicode(f):

    try:
        op.exists(u'\u201a')
    except UnicodeError:
        return None
    else:
        return f

class TestFile(TestCasePlus):

    def setUp(self):
        self.fname = res.get_data_copy("smpl_compound_chunked.hdf5")

    def tearDown(self):
        res.clear()

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

    def test_attr_dictcompat(self):
        # Test dictionary interface

        f = h5py.File(res.get_name(), 'w')

        values = {'a': 42, 'b': "hello", 'c': -32}
    
        self.assertRaises(KeyError, f.attrs.__getitem__, 'foo')
        self.assert_(not 'foo' in f.attrs)

        for key, val in values.iteritems():
            f.attrs[key] = val
            self.assert_(key in f.attrs)

        self.assertEqual(len(values), len(f.attrs))

        def assert_cmp_equal(what):
            c1 = getattr(f.attrs, what)
            c2 = getattr(values, what)
            self.assertEqual(set(c1()), set(c2()), what)

        for w in ['keys', 'values', 'items', 'iterkeys', 'itervalues', 'iteritems']:
            assert_cmp_equal(w)

        self.assertEqual(f.attrs.get('a', 99), 42)
        self.assertEqual(f.attrs.get('foo', 99), 99)
    
        del f.attrs['a']
        self.assert_(not 'a' in f.attrs)

class TestDataset(TestCasePlus):

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
                msg = "Extending %s to %s" % (shape, final_shape)
                newarr = numpy.arange(numpy.product(final_shape)).reshape(final_shape)
                ds.resize(final_shape)
                ds[...] = newarr
                self.assertEqual(ds.shape, final_shape, msg)
                self.assertArrayEqual(ds[...], newarr, msg)

            for illegal_shape in illegal_shapes[shape]:
                self.assertRaises(ValueError, ds.resize, illegal_shape)

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

        s = SliceFreezer()

        bases = [1024, 2**37, 2**60]
        shapes = [ (42,1), (100,100), (1,42), (1,1), (4,1025)]

        for base in bases:
            slices = [ s[base:base+x, base:base+y] for x, y in shapes]

            if "dset" in self.f:
                del self.f["dset"]

            dset = self.f.create_dataset("dset", (2**62, 2**62), '=f4', maxshape=(None,None))

            for shp, slc in zip(shapes, slices):
                msg = "Testing slice base 2**%d" % numpy.log2(base)

                empty = numpy.zeros(shp, dtype='=f4')
                data = numpy.arange(numpy.product(shp), dtype='=f4').reshape(shp)

                dset[slc] = empty
                arr = dset[slc]
                self.assertEqual(arr.shape, shp)
                self.assertArrayEqual(arr, empty, msg)
                
                dset[slc] = data
                arr = dset[slc]
                self.assertArrayEqual(arr, data, msg)

    def test_Dataset_exceptions(self):
        """ Test exceptions """
        # These trigger exceptions in H5Dread
        ref = numpy.ones((10,10), dtype='<i4')
        dsid = self.f.create_dataset('ds', data=ref)
        arr = numpy.ndarray((10,10), dtype='|S6') # incompatible datatype
        self.assertRaises(TypeError, dsid.id.read, h5s.ALL, h5s.ALL, arr)
        # or it'll segfault...

class TestExceptions(TestCasePlus):

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')

    def tearDown(self):
        try:
            self.f.close()
        except Exception:
            pass
        try:
            os.unlink(self.fname)
        except Exception:
            pass

    def test_groups(self):

        self.assertRaises(KeyError, self.f.__getitem__, "foobar")
        self.f.create_group("foobar")
        self.assertRaises(ValueError, self.f.create_group, "foobar")

    def test_attributes(self):
        
        g = self.f.create_group("foobar")
        self.assertRaises(KeyError, g.attrs.__getitem__, "attr")

class TestTypes(TestCasePlus):

    def setUp(self):
        pass

    def tearDown(self):
        res.clear()

    def test_enum(self):
        # Test high-level enumerated type

        vals = {'a': 1, 'b': 2, 'c': 42}
        
        f = h5py.File(res.get_name(), 'w')
        for idx, basetype in enumerate(np.dtype(x) for x in (common.INTS + common.UINTS)):

            msg = "dset %s, type %s" % (idx, basetype)

            dt = h5py.special_dtype(enum=(basetype, vals))
            self.assertEqual(h5py.check_dtype(enum=dt), vals, msg)
            self.assert_(h5py.check_dtype(enum=np.dtype('i')) is None, msg)

            # Test dataset creation
            refarr = np.zeros((4,4), dtype=dt)
            ds = f.create_dataset(str(idx), (4,4), dtype=dt)
            self.assert_(np.all(ds[...] == refarr), msg)

            # Test conversion to/from plain integer
            ds[0,0] = np.array(64, dtype=dt)
            self.assertEqual(ds[0,0], 64, msg)












