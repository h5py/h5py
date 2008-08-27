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
from common import getfullpath

class SliceFreezer(object):
    def __getitem__(self, args):
        return args


HDFNAME = getfullpath("smpl_compound_chunked.hdf5")

TYPES1 = \
  [ "<i1", "<i2", "<i4", "<i8", ">i1", ">i2", ">i4", ">i8", "|i1", "|u1", 
    "<u1", "<u2", "<u4", "<u8", ">u1", ">u2", ">u4", ">u8",
    "<f4", "<f8", ">f4", ">f8", "<c8", "<c16", ">c8", ">c16"]

TYPES2 = ["|S1", "|S2", "|S33", "|V1", "|V2", "|V33"]


TYPES = TYPES1 + TYPES2

SHAPES = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]

 
class TestFile(unittest.TestCase):

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

    def test_File_str(self):
        f = File(self.fname, 'r')
        str(f)
        f.close()
        str(f)

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

class TestDataset(unittest.TestCase):

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)

    def test_Dataset_create(self):
        
        print ''

        shapes = [(), (1,), (10,5), (1,10), (10,1), (100,1,100), (51,2,1025)]
        chunks = [None, (1,), (10,1), (1,1),  (1,1),  (50,1,100), (51,2,25)]

        # Test auto-chunk creation for each
        shapes += shapes
        chunks += [None]*len(chunks)

        for shape, chunk in zip(shapes, chunks):
            for dt in TYPES:
                print "    Creating %.20s %.40s" % (shape, dt)
                dt = numpy.dtype(dt)
                d = Dataset(self.f, "NewDataset", dtype=dt, shape=shape)
                self.assertEqual(d.shape, shape)
                self.assertEqual(d.dtype, dt)
                del self.f["NewDataset"]

                if shape != ():
                    print "        With chunk %s" % (chunk,)
                    d = Dataset(self.f, "NewDataset", dtype=dt, shape=shape,
                                chunks=chunk, shuffle=True, compression=6,
                                fletcher32=True)
                    self.assertEqual(d.shape, shape)
                    self.assertEqual(d.dtype, dt)
                    del self.f["NewDataset"]
             
                if 'V' not in dt.kind:
                    srcarr = numpy.ones(shape, dtype=dt)
                    d = Dataset(self.f, "NewDataset", data=srcarr)
                    self.assertEqual(d.shape, shape)
                    self.assertEqual(d.dtype, dt)
                    self.assert_(numpy.all(d.value == srcarr))
                    del self.f["NewDataset"]               

    def test_Dataset_extend(self):

        print ""

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
                print "    Extending %s to %s" % (shape, final_shape)
                newarr = numpy.arange(numpy.product(final_shape)).reshape(final_shape)
                ds.extend(final_shape)
                ds[...] = newarr
                self.assertEqual(ds.shape, final_shape)
                self.assert_(numpy.all(ds[...] == newarr))

            for illegal_shape in illegal_shapes[shape]:
                self.assertRaises(H5Error, ds.extend, illegal_shape)
        
    def test_Dataset_len_iter(self):

        arr1 = numpy.arange(100).reshape((10,10))
        arr2 = numpy.ones(())

        d1 = self.f.create_dataset("D1", data=arr1)
        d2 = self.f.create_dataset("D2", data=arr2)

        self.assertEqual(len(arr1), len(d1))
        self.assertRaises(TypeError, d2, len)

        for idx, (hval, nval) in enumerate(zip(d1, arr1)):
            self.assert_(numpy.all(hval == nval))
        
        self.assertEqual(idx+1, len(arr1))
        self.assertRaises(TypeError, list, d2)

    def test_Dataset_bigslice(self):
        print ""

        s = SliceFreezer()

        bases = [1024, 2**37, 2**60]
        shapes = [ (42,1), (100,100), (1,42), (1,1), (4,1025)]

        for base in bases:
            slices = [ s[base:base+x, base:base+y] for x, y in shapes]

            if "dset" in self.f:
                del self.f["dset"]

            dset = self.f.create_dataset("dset", (2**62, 2**62), '=f4', maxshape=(None,None))

            for shp, slc in zip(shapes, slices):
                print "    Testing base 2**%d" % numpy.log2(base)

                empty = numpy.zeros(shp)
                data = numpy.arange(numpy.product(shp), dtype='=f4').reshape(shp)

                dset[slc] = empty
                arr = dset[slc]
                self.assertEqual(arr.shape, shp)
                self.assert_(numpy.all(arr == empty), "%r \n\n %r" % (arr, empty))
                
                dset[slc] = data
                arr = dset[slc]
                self.assert_(numpy.all(arr == data), "%r \n\n %r" % (arr, data))
        
    def test_Dataset_slicing(self):

        print ''

        s = SliceFreezer()
        slices = [s[0,0,0], s[0,0,:], s[0,:,0], s[0,:,:]]
        slices += [s[0:1,:,4:5], s[2:3,0,4:5], s[:,0,0:1], s[0,:,0:1]]
        slices += [ s[9,9,49], s[9,:,49], s[9,:,:] ]
        slices += [ s[0, ..., 49], s[...], s[..., 49], s[9,...] ]
        slices += [ s[0:7:2,0:9:3,15:43:5], s[2:8:2,...] ]
        slices += [ s[0], s[1], s[9], s[0,0], s[4,5], s[:] ]
        slices += [ s[3,...], s[3,2,...] ]
        slices += [ numpy.random.random((10,10,50)) > 0.5 ]  # Truth array
        for dt in TYPES1:

            srcarr = numpy.arange(10*10*50, dtype=dt).reshape(10,10,50)
            srcarr = srcarr + numpy.sin(srcarr)


            fname = tempfile.mktemp('.hdf5')
            f = File(fname, 'w')
            try:
                d = Dataset(f, "NewDataset", data=srcarr)
                self.assertEqual(d.shape, srcarr.shape)
                self.assertEqual(d.dtype, srcarr.dtype)
                for argtpl in slices:
                    # Test read
                    print "    Checking read %.20s %s" % (dt, argtpl if not isinstance(argtpl, numpy.ndarray) else 'ARRAY')
                    hresult = d[argtpl]
                    nresult = srcarr[argtpl]
                    if isinstance(nresult, numpy.ndarray):
                        self.assertEqual(hresult.shape, nresult.shape)
                        self.assertEqual(hresult.dtype, nresult.dtype)
                    else:
                        self.assert_(not isinstance(hresult, numpy.ndarray))
                    self.assert_(numpy.all(hresult == nresult))

                del f["NewDataset"]
                d = Dataset(f, "NewDataset", data=srcarr)
                for argtpl in slices:
                    # Test assignment
                    print "    Checking write %.20s %s" % (dt, argtpl if not isinstance(argtpl, numpy.ndarray) else 'ARRAY')
                    srcarr[argtpl] = numpy.cos(srcarr[argtpl])
                    d[argtpl] = srcarr[argtpl]
                    self.assert_(numpy.all(d.value == srcarr))
                    
            finally:
                f.close()
                os.unlink(fname)   


    def test_Dataset_exceptions(self):
        # These trigger exceptions in H5Dread
        ref = numpy.ones((10,10), dtype='<i4')
        dsid = self.f.create_dataset('ds', data=ref)
        arr = numpy.ndarray((10,10), dtype='|S6') # incompatible datatype
        self.assertRaises(H5Error, dsid.id.read, h5s.ALL, h5s.ALL, arr)
        # or it'll segfault...

class TestGroup(unittest.TestCase):

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)

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
        self.assertEqual(set(self.f), set(subgroups))

        # iteritems()
        for name, obj in self.f.iteritems():
            self.assert_(name in subgroups)
            self.assert_(isinstance(obj, Group))

        # __delitem__
        for name in subgroups:
            self.assert_(name in self.f)
            del self.f[name]
            self.assert_(not name in self.f)

        self.assertEqual(len(self.f), 0)

        # __str__
        grp = self.f.create_group("Foobar")
        str(grp)
        grp.id._close()
        str(grp)

    def test_Group_setgetitem(self):
        # Also tests named types

        print ''
        for shape in SHAPES:
            for dt in TYPES1:

                print "    Assigning %s %s" % (dt, shape)

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

        

        
        
        

        

        


















