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
import tempfile
import os
from numpy import dtype

from h5py import *
from h5py import h5t, h5f
from common import TestCasePlus, api_18, res
import cPickle

typecode_map = {'i': h5t.INTEGER, 'u': h5t.INTEGER, 'f': h5t.FLOAT,
           'c': h5t.COMPOUND, 'S': h5t.STRING, 'V': h5t.OPAQUE}

simple_types = \
  [ "<i1", "<i2", "<i4", "<i8", ">i1", ">i2", ">i4", ">i8", "|i1", "|u1", 
    "<u1", "<u2", "<u4", "<u8", ">u1", ">u2", ">u4", ">u8",
    "<f4", "<f8", ">f4", ">f8", "<c8", "<c16", ">c8", ">c16",
    "|S1", "|S2", "|S33", "|V1", "|V2", "|V33"]

def mkstr(arr):
    return "".join(chr(x) for x in arr)

class BaseTypeMixin(object):

    """
        Base class for TypeID tests, which tests that the various TypeID
        subclasses correctly implement operations like equality and
        serialization.

        MUST be a mixin or the stupid unittest loader tries to test it.
    """

    def tearDown(self):
        res.clear()

    ODDBALL_TYPE = h5t.create(h5t.OPAQUE, 72)

    # --- The following attributes and methods MUST be overridden ---

    CLASSCODE = -1

    def get_example_type(self):
        """ Override this method to provide a dynamically-created example
         type, which will be used to test common TypeID operations.
        """
        raise NotImplementedError("!")

    def get_example_encoding(self):
        """ Override this method to provide a known-good serialized example
        of the example type above.  Only called with 1.8 API.
        """
        raise NotImplementedError("!")

    # --- End required attributes and methods ---

    @api_18
    def test_serialize(self):
        """ Generic subtype serialization test
        """
        # Round-trip serialization
        htype = self.get_example_type()
        htype2 = h5t.decode(htype.encode())
        self.assertEqual(htype, htype2)

        # Deserialization of known buffer
        ser = self.get_example_encoding()
        htype3 = h5t.decode(ser)
        self.assertEqual(htype, htype2)

        # Pickling
        pkl = cPickle.dumps(htype)
        htype4 = cPickle.loads(pkl)
        self.assertEqual(htype, htype4)

    def test_commit(self):
        """ Generic subtype commit test
        """
        fid = h5f.create(res.get_name())
        htype = self.get_example_type()
        self.assert_(not htype.committed())
        htype.commit(fid, "name")
        self.assert_(htype.committed())
        htype2 = h5t.open(fid, "name")
        self.assertEqual(htype, htype2)

class TestInteger(TestCasePlus, BaseTypeMixin):

    CLASSCODE = h5t.INTEGER

    def get_example_type(self):
        return h5t.STD_I32LE.copy()
    def get_example_encoding(self):
        return mkstr([3, 0, 16, 8, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0])

    def test_set_get_order_sign(self):
        
        htype = h5t.STD_I32LE.copy()

        self.assertEqual(htype.get_order(), h5t.ORDER_LE)
        self.assertEqual(htype.get_sign(), h5t.SGN_2)

        htype.set_order(h5t.ORDER_BE)
        htype.set_sign(h5t.SGN_NONE)
        
        self.assertEqual(htype.get_order(), h5t.ORDER_BE)
        self.assertEqual(htype.get_sign(), h5t.SGN_NONE)

class TestFloat(TestCasePlus, BaseTypeMixin):

    CLASSCODE = h5t.FLOAT   

    def get_example_type(self):
        return h5t.IEEE_F32LE.copy()
    def get_example_encoding(self):
        return mkstr([3, 0, 17, 32, 31, 0, 4, 0, 0, 0, 0, 0, 
                      32, 0, 23, 8, 0, 23, 127, 0, 0, 0])

class TestString(TestCasePlus, BaseTypeMixin):

    CLASSCODE = h5t.STRING

    def get_example_type(self):
        return h5t.C_S1.copy()
    def get_example_encoding(self):
        return mkstr([3, 0, 19, 0, 0, 0, 1, 0, 0, 0])

class TestOpaque(TestCasePlus, BaseTypeMixin):

    CLASSCODE = h5t.OPAQUE

    def get_example_type(self):
        return h5t.create(h5t.OPAQUE, 31)
    def get_example_encoding(self):
        return mkstr([3, 0, 21, 0, 0, 0, 31, 0, 0, 0])

    def test_setget_tag(self):
        htype = h5t.create(h5t.OPAQUE, 40)
        htype.set_tag("FOOBAR")
        self.assertEqual(htype.get_tag(), "FOOBAR")


class TestH5T(TestCasePlus):

    def test_get_set_size(self):

        sizes = (1,2,3,4,127,128,129,133,16385)

        def test(htype, size):
            htype.set_size(size)
            self.assertEqual(htype.get_size(), size)  

        htype = h5t.create(h5t.OPAQUE, 4)
        for size in sizes:
            test(htype, size)

    def test_get_super(self):

        def test(dt):
            """ Check get_super for a given dtype """
            htype = h5t.py_create(dt)
            atype = h5t.array_create(htype, (4,5))
            self.assert_(htype.equal(atype.get_super()))

        for x in simple_types:
            test(x)

    def test_detect_class(self):
        
        dt = dtype([(x, x) for x in simple_types])

        htype = h5t.py_create(dt)
        self.assert_(htype.detect_class(h5t.INTEGER))
        self.assert_(htype.detect_class(h5t.OPAQUE))
        self.assert_(not htype.detect_class(h5t.ARRAY))


        
    def test_array(self):
        """ Test all array-specific features """
        htype = h5t.array_create(h5t.STD_I32LE,(4,5))
        self.assertEqual(htype.get_array_ndims(), 2)
        self.assertEqual(htype.get_array_dims(), (4,5))
        self.assertEqual(htype.dtype, dtype(('<i4',(4,5))))

    def test_enum(self):
        """ Test all enum-specific routines """
        names = ("A", "B", "Name3", "Name with space", " 12A-d878dd&%2 0-1!** ")
        values = (1,2,3.0, -999, 30004.0)
        valuedict = {}
        htype = h5t.enum_create(h5t.STD_I32LE)

        for idx, (name, value) in enumerate(zip(names, values)):
            htype.enum_insert(name, value)
            valuedict[name] = value
            self.assertEqual(htype.get_member_value(idx), value)

        for name, value in valuedict.iteritems():
            self.assertEqual(htype.enum_valueof(name), value)
            self.assertEqual(htype.enum_nameof(value), name)

        self.assertEqual(htype.get_nmembers(), len(names))

    def test_compound(self):
        """ Test all compound datatype operations """
        names = ("A", "B", "Name3", "Name with space", " 12A-d878dd&%2 0-1!** ")
        types = (h5t.STD_I8LE, h5t.IEEE_F32BE, h5t.STD_U16BE, h5t.C_S1.copy(), h5t.FORTRAN_S1.copy())
        types[3].set_size(8)
        types[4].set_size(8)
        classcodes = (h5t.INTEGER, h5t.FLOAT, h5t.INTEGER, h5t.STRING, h5t.STRING)
        
        # Align all on 128-bit (16-byte) boundaries
        offsets = tuple(x*16 for x in xrange(len(names)))
        total_len = 16*len(names)
        htype = h5t.create(h5t.COMPOUND, total_len)

        for idx, name in enumerate(names):
            htype.insert(name, offsets[idx], types[idx])
        
        for idx, name in enumerate(names):
            self.assertEqual(htype.get_member_name(idx), name)
            self.assertEqual(htype.get_member_class(idx), classcodes[idx])
            self.assertEqual(htype.get_member_index(name), idx)
            self.assertEqual(htype.get_member_offset(idx), offsets[idx])
            self.assert_(htype.get_member_type(idx).equal(types[idx]))

        self.assertEqual(htype.get_size(), total_len)
        htype.pack()
        self.assert_(htype.get_size() < total_len)
        self.assertEqual(htype.get_nmembers(), len(names))

    def test_bool(self):
        out = h5t.py_create('bool')
        self.assert_(isinstance(out, h5t.TypeEnumID))
        self.assertEqual(out.get_nmembers(),2)
        self.assertEqual(out.dtype, dtype('bool'))

    # === Tests for py_create =================================================

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
        config = h5.get_config()

        oldnames = config.complex_names
        try:
            for name in names:
                config.complex_names = name
                for ctype in complex_types:
                    dt = dtype(ctype)
                    htype = h5t.py_create(dt)
                    self.assertEqual(type(htype), h5t.TypeCompoundID)
                    self.assertEqual(htype.get_nmembers(), 2)
                    self.assertEqual(htype.get_member_name(0), name[0])
                    self.assertEqual(htype.get_member_name(1), name[1])
        finally:
            config.complex_names = oldnames

import h5py
cfg = h5py.get_config()
bytemap = {'<': h5t.ORDER_LE, '>': h5t.ORDER_BE, '=': h5t.ORDER_NATIVE}

class TestPyCreate(TestCasePlus):

    """
        Tests the translation from Python dtypes to HDF5 datatypes
    """

    def test_integer(self):
        """ Signed integer translation

        - TypeIntegerID
        - 1, 2, 4, 8 bytes
        - LE and BE
        - Signed
        """
        bases = ('=i', '<i', '>i')

        for b in bases:
            for l in (1, 2, 4, 8):
                dt = '%s%s' % (b, l)
                htype = h5t.py_create(dt)
                self.assert_(isinstance(htype, h5t.TypeIntegerID), "wrong class")
                self.assertEqual(htype.get_size(), l, "wrong size")
                self.assertEqual(htype.get_sign(), h5t.SGN_2, "wrong sign")
                if l != 1:  # NumPy does not allow ordering of 1-byte types
                    self.assertEqual(htype.get_order(), bytemap[b[0]])

    def test_uinteger(self):
        """ Unsigned integer translation

        - TypeIntegerID
        - 1, 2, 4, 8 bytes
        - LE and BE
        - Unsigned
        """
        bases = ('=u', '<u', '>u')

        for b in bases:
            for l in (1, 2, 4, 8):
                dt = "%s%s" % (b, l)
                htype = h5t.py_create(dt)
                self.assert_(isinstance(htype, h5t.TypeIntegerID), "wrong class")
                self.assertEqual(htype.get_size(), l, "wrong size")
                self.assertEqual(htype.get_sign(), h5t.SGN_NONE, "wrong sign")
                if l != 1:
                    self.assertEqual(htype.get_order(), bytemap[b[0]], "wrong order")

    def test_float(self):
        """ Floating-point translation

        - TypeFloatID
        - 1, 2, 4, 8 bytes
        - LE and BE
        - Unsigned
        """
        bases = ('=f', '<f', '>f')
        
        for b in bases:
            for l in (4, 8):
                dt = "%s%s" % (b, l)
                htype = h5t.py_create(dt)
                self.assert_(isinstance(htype, h5t.TypeFloatID), "wrong class")
                self.assertEqual(htype.get_size(), l, "wrong size")
                self.assertEqual(htype.get_order(), bytemap[b[0]])

    def test_complex(self):
        """ Complex type translation

        - TypeComplexID
        - 8, 16 bytes
        - LE and BE
        - 2 members
        - Member names from cfg.complex_names
        - Members are TypeFloatID
        """
        bases = ('=c', '<c', '>c')
        
        for b in bases:
            for l in (8, 16):
                dt = '%s%s' % (b, l)
                htype = h5t.py_create(dt)
                self.assert_(isinstance(htype, h5t.TypeCompoundID), "wrong class")
                self.assertEqual(htype.get_size(), l, "wrong size")
                self.assertEqual(htype.get_nmembers(), 2, "wrong # members")
                for idx in (0, 1):
                    self.assertEqual(htype.get_member_name(idx), cfg.complex_names[idx])
                    st = htype.get_member_type(idx)
                    self.assert_(isinstance(st, h5t.TypeFloatID))
                    self.assertEqual(st.get_size(), l//2)
                    self.assertEqual(st.get_order(), bytemap[b[0]])

    def test_string(self):
        """ Fixed-length string translation

        - TypeStringID
        - Fixed-length
        - Size 1 byte to 2**31-1 bytes
        - Charset ASCII
        - Null-padded
        """

        for l in (1, 23, 2**31-1):
            dt = '|S%s' % l
            htype = h5t.py_create(dt)
            self.assert_(isinstance(htype, h5t.TypeStringID), "wrong class")
            self.assertEqual(htype.get_size(), l)
            self.assertEqual(htype.get_cset(), h5t.CSET_ASCII, "wrong cset")
            self.assertEqual(htype.get_strpad(), h5t.STR_NULLPAD, "wrong padding")
            self.assert_(not htype.is_variable_str(), "should be fixed str")

    def test_vlstring(self):
        """ Variable-length string translation

        In literal mode:
        - TypeOpaqueID
        - Equal to PYTHON_OBJECT

        In logical mode:
        - TypeStringID
        - Variable-length
        - Charset ASCII
        - Null-terminated
        """

        dt = h5t.py_new_vlen(str)
        htype = h5t.py_create(dt)
        self.assert_(isinstance(htype, h5t.TypeOpaqueID))
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

        htype = h5t.py_create(dt, logical=True)
        self.assert_(isinstance(htype, h5t.TypeStringID))
        self.assert_(htype.is_variable_str())
        self.assertEqual(htype.get_cset(), h5t.CSET_ASCII)
        self.assertEqual(htype.get_strpad(), h5t.STR_NULLTERM)

    def test_boolean(self):
        """ Boolean type translation

        - TypeEnumID
        - Base TypeIntegerID
        - Base 1 byte
        - Base signed
        - Member names from cfg.bool_names
        - 2 values
        - Values 0, 1
        """

        htype = h5t.py_create('bool')
        self.assert_(isinstance(htype, h5t.TypeEnumID), "wrong class")
        self.assertEqual(htype.get_nmembers(), 2, "must be 2-element enum")
        basetype = htype.get_super()
        self.assertEqual(basetype.get_size(), 1, "wrong size")
        self.assertEqual(basetype.get_sign(), h5t.SGN_2, "wrong sign")
        for idx in (0,1):
            self.assertEqual(htype.get_member_name(idx), cfg.bool_names[idx], "wrong name")
            self.assertEqual(htype.get_member_value(idx), idx, "wrong value")

    def test_opaque(self):
        """ Opaque type translation

        - TypeOpaqueID
        - Sizes 1 byte to 2**31-1 bytes
        - Empty tag
        """

        for l in (1, 21, 2**31-1):
            htype = h5t.py_create('|V%s' % l)
            self.assert_(isinstance(htype, h5t.TypeOpaqueID))
            self.assertEqual(htype.get_size(), l)
            self.assertEqual(htype.get_tag(), "")

    def test_enum(self):
        """ Enum type translation

        Literal:
        - TypeIntegerID

        Logical:
        - TypeEnumID
        - Base TypeIntegerID
        - 0 to (at least) 128 values
        """
        enums = [{}, {'a': 0, 'b': 1}, dict(("%s" % d, d) for d in xrange(128)) ]
        bases = ('|i1', '|u1', '<i4', '>i4', '<u8')

        for b in bases:
            for e in enums:
                dt = h5t.py_new_enum(b, e)
                htype_comp = h5t.py_create(b)
                htype = h5t.py_create(dt)
                self.assert_(isinstance(htype, h5t.TypeIntegerID))
                self.assertEqual(htype, htype_comp)
                htype = h5t.py_create(dt, logical=True)
                self.assert_(isinstance(htype, h5t.TypeEnumID), "%s" % (htype,))
                basetype = htype.get_super()
                self.assertEqual(htype_comp, basetype)
                self.assertEqual(htype.get_nmembers(), len(e))
                for idx in xrange(htype.get_nmembers()):
                    name = htype.get_member_name(idx)
                    value = htype.get_member_value(idx)
                    self.assertEqual(e[name], value)




