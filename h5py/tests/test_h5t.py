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

import unittest
from numpy import dtype

import h5py
from h5py import h5t, h5i
from h5py.h5 import H5Error

kind_map = {'i': h5t.TypeIntegerID, 'u': h5t.TypeIntegerID, 'f': h5t.TypeFloatID,
           'c': h5t.TypeCompoundID, 'S': h5t.TypeStringID, 'V': h5t.TypeOpaqueID}

typecode_map = {'i': h5t.INTEGER, 'u': h5t.INTEGER, 'f': h5t.FLOAT,
           'c': h5t.COMPOUND, 'S': h5t.STRING, 'V': h5t.OPAQUE}

simple_types = \
  [ "<i1", "<i2", "<i4", "<i8", ">i1", ">i2", ">i4", ">i8", "|i1", "|u1", 
    "<u1", "<u2", "<u4", "<u8", ">u1", ">u2", ">u4", ">u8",
    "<f4", "<f8", ">f4", ">f8", "<c8", "<c16", ">c8", ">c16",
    "|S1", "|S2", "|S33", "|V1", "|V2", "|V33"]

class TestH5T(unittest.TestCase):

    def tearDown(self):
        h5t.py_complex_names(reset=True)

    def test_create(self):
        types = {h5t.COMPOUND: h5t.TypeCompoundID, h5t.OPAQUE: h5t.TypeOpaqueID}
        sizes = (1,4,256)
        for typecode, typeobj in types.iteritems():
            for size in sizes:
                htype = h5t.create(typecode, size)
                self.assertEqual(type(htype), typeobj)
                self.assertEqual(htype.get_size(), size)
        
        self.assertRaises(ValueError, h5t.create, h5t.ARRAY, 4)
    
    def test_copy(self):

        for x in simple_types:
            htype = h5t.py_create(dtype(x))
            htype2 = htype.copy()
            self.assertEqual(htype.dtype, htype2.dtype)
            self.assert_(htype is not htype2)
            self.assert_(htype != htype2)

    def test_equal(self):

        htype = h5t.create(h5t.OPAQUE, 128)
        htype2 = h5t.create(h5t.OPAQUE, 128)
        htype3 = h5t.create(h5t.OPAQUE, 127)

        self.assert_(htype.equal(htype2))
        self.assert_(not htype.equal(htype3))

    def test_lock(self):

        htype = h5t.STD_I8LE.copy()
        htype.set_sign(h5t.SGN_NONE)
        htype.lock()
        self.assertRaises(H5Error, htype.set_sign, h5t.SGN_2)

    def test_get_class(self):

        for x in simple_types:
            dt = dtype(x)
            htype = h5t.py_create(dt)
            self.assertEqual(htype.get_class(), typecode_map[dt.kind])

    def test_get_size(self):

        sizes = (1,2,3,4,127,128,129,133,16385)
        for x in sizes:
            htype = h5t.create(h5t.OPAQUE, x)
            self.assertEqual(htype.get_size(), x)

    def test_get_super(self):

        for x in simple_types:
            htype = h5t.py_create(x)
            atype = h5t.array_create(htype, (4,5))
            self.assert_(htype.equal(atype.get_super()))

    def test_detect_class(self):

        dt = dtype([(x, x) for x in simple_types])

        htype = h5t.py_create(dt)
        self.assert_(htype.detect_class(h5t.INTEGER))
        self.assert_(htype.detect_class(h5t.OPAQUE))
        self.assert_(not htype.detect_class(h5t.ARRAY))

    def test_set_size(self):

        htype = h5t.create(h5t.OPAQUE, 128)
        self.assertEqual(htype.get_size(), 128)
        htype.set_size(300)
        self.assertEqual(htype.get_size(), 300)

    def test_set_get_order_sign(self):
        
        htype = h5t.STD_I32LE.copy()

        self.assertEqual(htype.get_order(), h5t.ORDER_LE)
        self.assertEqual(htype.get_sign(), h5t.SGN_2)

        htype.set_order(h5t.ORDER_BE)
        htype.set_sign(h5t.SGN_NONE)
        
        self.assertEqual(htype.get_order(), h5t.ORDER_BE)
        self.assertEqual(htype.get_sign(), h5t.SGN_NONE)

    
    # === Tests for py_create =================================================

    def test_py_create_simple(self):

        for x in simple_types:
            dt = dtype(x)
            htype = h5t.py_create(dt)
            self.assertEqual(type(htype), kind_map[dt.kind])
            self.assertEqual(dt, htype.dtype)

    def test_py_create_enum(self):
        enum = {'A': 0, 'AA': 1, 'foo': 34, 'bizarre': 127}
        enum_bases = [ x for x in simple_types if 'i' in x or 'u' in x]
        for x in enum_bases:
            dt = dtype(x)
            htype = h5t.py_create(dt, enum=enum)
            self.assertEqual(type(htype), h5t.TypeEnumID)
            self.assertEqual(dt, htype.dtype)
            for name, val in enum.iteritems():
                self.assertEqual(name, htype.enum_nameof(val))

    def test_py_create_array(self):
        shapes = [ (1,1), (1,), (4,5), (99,10,22) ]
        array_types = []
        for base in simple_types:
            for shape in shapes:
                array_types.append((base, shape))

        for x in array_types:
            dt = dtype(x)
            htype = h5t.py_create(dt)
            self.assertEqual(type(htype), h5t.TypeArrayID)
            self.assertEqual(dt, htype.dtype)

    def test_py_create_compound(self):

        # Compound type, each field of which is named for its type
        simple_compound = [ (x, x) for x in simple_types ]
        deep_compound = [ ('A', simple_compound), ('B', '<i4') ]

        compound_types = [simple_compound, deep_compound]
        for x in compound_types:
            dt = dtype(x)
            htype = h5t.py_create(dt)
            self.assertEqual(type(htype), h5t.TypeCompoundID)
            self.assertEqual(dt, htype.dtype)

    def test_names(self):

        names = [('r','i'), ('real', 'imag'), (' real name ', ' img name '),
                 (' Re!#@$%\t\tREALr\neal ^;;;"<>? ', ' \t*&^  . ^@IMGI        MG!~\t\n\r') ]

        complex_types = [x for x in simple_types if 'c' in x]

        try:
            for name in names:
                h5t.py_complex_names(name[0], name[1])
                for ctype in complex_types:
                    dt = dtype(ctype)
                    htype = h5t.py_create(dt)
                    self.assertEqual(type(htype), h5t.TypeCompoundID)
                    self.assertEqual(htype.get_nmembers(), 2)
                    self.assertEqual(htype.get_member_name(0), name[0])
                    self.assertEqual(htype.get_member_name(1), name[1])
        finally:
            h5t.py_complex_names(reset=True)




