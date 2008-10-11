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

from numpy import array, ndarray, dtype, all, ones

from common import TestBase

from h5py import *
from h5py.h5 import H5Error

# === attributes.hdf5 description ===

HDFNAME = 'attributes.hdf5'
OBJECTNAME = 'Group'
ATTRIBUTES = {  'String Attribute': ("This is a string.", dtype('S18'), ()),
                'Integer': (42, dtype('<i4'), ()),
                'Integer Array': ( [0,1,2,3], dtype('<i4'), (4,) ),
                'Byte': (-34, dtype('|i1'), ()) }
ATTRIBUTES_ORDER = ['String Attribute', 'Integer', 'Integer Array', 'Byte']
NEW_ATTRIBUTES = {'New float': ( 3.14, dtype('<f4'), ()) }

class TestH5A(TestBase):

    HDFNAME = HDFNAME

    def setUp(self):
        TestBase.setUp(self)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        self.obj._close()
        TestBase.tearDown(self)

    def is_attr(self, attr):
        return (h5i.get_type(attr) == h5i.ATTR)

    # === General attribute operations ========================================

    def test_create_write(self):

        obj = h5g.open(self.fid, OBJECTNAME)
        for name, (value, dt, shape) in NEW_ATTRIBUTES.iteritems():
            arr_ref = array(value, dtype=dt)
            arr_fail = ones((15,15), dtype=dt)

            space = h5s.create(h5s.SCALAR)
            htype = h5t.py_create(dt)

            attr = h5a.create(obj, name, htype, space)
            self.assert_(self.is_attr(attr))
            attr.write(arr_ref)
            self.assertRaises(TypeError, attr.write, arr_fail)

            attr = h5a.open(obj, name)
            dt = attr.dtype
            shape = attr.shape
            arr_val = ndarray(shape, dtype=dt)
            attr.read(arr_val)
            self.assert_(all(arr_val == arr_ref))

    def test_open_idx(self):
        for idx, name in enumerate(ATTRIBUTES_ORDER):
            attr = h5a.open(self.obj, idx=idx)
            self.assert_(self.is_attr(attr), "Open: index %d" % idx)

    def test_open(self):
        for name in ATTRIBUTES:
            attr = h5a.open(self.obj, name)
            self.assert_(self.is_attr(attr), 'Open: name "%s"' % name)

    def test_close(self):
        attr = h5a.open(self.obj, idx=0)
        self.assert_(self.is_attr(attr))
        attr._close()
        self.assert_(not self.is_attr(attr))

    def test_delete(self):

        obj = h5g.open(self.fid, OBJECTNAME)

        attr = h5a.open(obj, ATTRIBUTES_ORDER[0])
        self.assert_(self.is_attr(attr))
        del attr

        h5a.delete(obj, ATTRIBUTES_ORDER[0])
        self.assertRaises(H5Error, h5a.open, obj, ATTRIBUTES_ORDER[0])


    # === Attribute I/O =======================================================

    def test_read(self):
        for name in ATTRIBUTES:
            value, dt, shape = ATTRIBUTES[name]

            attr = h5a.open(self.obj, name)
            arr_holder = ndarray(shape, dtype=dt)
            arr_reference = array(value, dtype=dt)

            self.assertEqual(attr.shape, shape)
            self.assertEqual(attr.dtype, dt)

            attr.read(arr_holder)
            self.assert_( all(arr_holder == arr_reference) )

    # write is done by test_create_write

    # === Attribute inspection ================================================

    def test_get_num_attrs(self):
        n = h5a.get_num_attrs(self.obj)
        self.assertEqual(n, len(ATTRIBUTES))

    def test_get_name(self):
    
        for name in ATTRIBUTES:
            attr = h5a.open(self.obj, name)
            self.assertEqual(attr.get_name(), name)

    def test_get_space(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open(self.obj, name)
            space = attr.get_space()
            shape_tpl = space.get_simple_extent_dims()
            self.assertEqual(shape_tpl, shape)

    def test_get_type(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open(self.obj, name)
            htype = attr.get_type()

    def test_iterate(self):

        def iterate_all(id, name, namelist):
            namelist.append(name)

        def iterate_two(id, name, namelist):
            if len(namelist) == 2:
                raise StopIteration
            namelist.append(name)

        def iterate_fault(id, name, namelist):
            if len(namelist) == 2:
                raise RuntimeError("Intentional fault")
            namelist.append(name)

        namelist = []
        h5a.iterate(self.obj, iterate_all, namelist)
        self.assertEqual(namelist, ATTRIBUTES_ORDER)

        namelist = []
        h5a.iterate(self.obj, iterate_two, namelist)
        self.assertEqual(namelist, ATTRIBUTES_ORDER[0:2])

        namelist = []
        self.assertRaises(RuntimeError, h5a.iterate, self.obj, iterate_fault, namelist)
        self.assertEqual(namelist, ATTRIBUTES_ORDER[0:2])
        
        namelist = []
        h5a.iterate(self.obj, iterate_two, namelist, 1)
        self.assertEqual(namelist, ATTRIBUTES_ORDER[1:3])

    def test_prop_name(self):
        
        for name in ATTRIBUTES:
            attr = h5a.open(self.obj, name)
            self.assertEqual(attr.name, name)

    def test_prop_shape(self):

        for name, (val, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open(self.obj, name)
            self.assertEqual(attr.shape, shape)

    def test_prop_dtype(self):

        for name, (val, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open(self.obj, name)
            self.assertEqual(attr.dtype, dt)

    def test_py_listattrs(self):

        attrlist = h5a.py_listattrs(self.obj)

        self.assertEqual(attrlist, ATTRIBUTES_ORDER)

    def test_py_exists(self):

        self.assert_(h5a.py_exists(self.obj, ATTRIBUTES_ORDER[0]))
        self.assert_(not h5a.py_exists(self.obj, "Something else"))









