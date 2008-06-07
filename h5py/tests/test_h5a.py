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
from numpy import array, ndarray, dtype, all, ones
import os

import h5py
from h5py import h5a
from h5py import h5f, h5g, h5i, h5t, h5s
from h5py.h5e import H5Error

from common import getcopy, deletecopy, errstr


HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/attributes.hdf5')
OBJECTNAME = 'Group'
ATTRIBUTES = {  'String Attribute': ("This is a string.", dtype('S18'), ()),
                'Integer': (42, dtype('<i4'), ()),
                'Integer Array': ( [0,1,2,3], dtype('<i4'), (4,) ),
                'Byte': (-34, dtype('|i1'), ()) }
ATTRIBUTES_ORDER = ['String Attribute', 'Integer', 'Integer Array', 'Byte']
NEW_ATTRIBUTES = {'New float': ( 3.14, dtype('<f4'), ()) }

class TestH5A(unittest.TestCase):

    def setUp(self):
        self.fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        h5g.close(self.obj)
        h5f.close(self.fid)

    def is_attr(self, aid):
        return (h5i.get_type(aid) == h5i.ATTR)

    # === General attribute operations ========================================

    def test_create_write(self):
        fid, filename = getcopy(HDFNAME)
        obj = h5g.open(fid, OBJECTNAME)
        for name, (value, dt, shape) in NEW_ATTRIBUTES.iteritems():
            arr_ref = array(value, dtype=dt)
            arr_fail = ones((15,15), dtype=dt)

            sid = h5s.create(h5s.SCALAR)
            tid = h5t.py_dtype_to_h5t(dt)

            aid = h5a.create(obj, name, tid, sid)
            self.assert_(self.is_attr(aid))
            h5a.write(aid, arr_ref)
            self.assertRaises(ValueError, h5a.write, aid, arr_fail)
            h5a.close(aid)

            arr_val = h5a.py_get(obj,name)
            self.assert_(all(arr_val == arr_ref), errstr(arr_val, arr_ref))
            h5s.close(sid)
        h5g.close(obj)
        deletecopy(fid, filename)
        
        self.assertRaises(TypeError, h5a.create, -1, "FOOBAR", -1, -1)
        self.assertRaises(TypeError, h5a.write, -1, arr_ref)

    def test_open_idx(self):
        for idx, name in enumerate(ATTRIBUTES_ORDER):
            aid = h5a.open_idx(self.obj, idx)
            self.assert_(self.is_attr(aid), "Open: index %d" % idx)
            h5a.close(aid)
    
        self.assertRaises(TypeError, h5a.open_idx, -1, 0)

    def test_open_name(self):
        for name in ATTRIBUTES:
            aid = h5a.open_name(self.obj, name)
            self.assert_(self.is_attr(aid), 'Open: name "%s"' % name)
            h5a.close(aid)

        self.assertRaises(TypeError, h5a.open_name, -1, "foo")

    def test_close(self):
        aid = h5a.open_idx(self.obj, 0)
        self.assert_(self.is_attr(aid))
        h5a.close(aid)
        self.assert_(not self.is_attr(aid))
    
        self.assertRaises(TypeError, h5a.close, -1)

    def test_delete(self):
        fid, filename = getcopy(HDFNAME)
        obj = h5g.open(fid, OBJECTNAME)
        self.assert_(h5a.py_exists(obj, ATTRIBUTES_ORDER[0]))
        h5a.delete(obj, ATTRIBUTES_ORDER[0])
        self.assert_(not h5a.py_exists(obj, ATTRIBUTES_ORDER[0]))
        deletecopy(fid, filename)

        self.assertRaises(TypeError, h5a.delete, -1, "foo")

    # === Attribute I/O =======================================================

    def test_read(self):
        for name in ATTRIBUTES:
            value, dt, shape = ATTRIBUTES[name]

            aid = h5a.open_name(self.obj, name)
            arr_holder = ndarray(shape, dtype=dt)
            arr_reference = array(value, dtype=dt)

            if len(shape) != 0:
                arr_fail = ndarray((), dtype=dt)
                self.assertRaises(ValueError, h5a.read, aid, arr_fail)

            h5a.read(aid, arr_holder)
            self.assert_( all(arr_holder == arr_reference),
                errstr(arr_reference, arr_holder, 'Attr "%s"):\n' % name, ))

            h5a.close(aid)
        
        self.assertRaises(TypeError, h5a.read, -1, arr_holder)

    # h5a.write is done by test_create_write

    # === Attribute inspection ================================================

    def test_get_num_attrs(self):
        n = h5a.get_num_attrs(self.obj)
        self.assertEqual(n, len(ATTRIBUTES))
        self.assertRaises(H5Error, h5a.get_num_attrs, -1)

    def test_get_name(self):
    
        for name in ATTRIBUTES:
            aid = h5a.open_name(self.obj, name)
            supposed_name = h5a.get_name(aid)
            self.assertEqual(supposed_name, name)
            h5a.close(aid)

        self.assertRaises(TypeError, h5a.get_name, -1)

    def test_get_space(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            aid = h5a.open_name(self.obj, name)
            sid = h5a.get_space(aid)
            shape_tpl = h5s.get_simple_extent_dims(sid)
            self.assertEqual(shape_tpl, shape)
            h5s.close(sid)
            h5a.close(aid)

        self.assertRaises(TypeError, h5a.get_space, -1)

    def test_get_type(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            aid = h5a.open_name(self.obj, name)
            tid = h5a.get_type(aid)
            supposed_dtype = h5t.py_h5t_to_dtype(tid)
            self.assertEqual(supposed_dtype, dt)
            h5t.close(tid)
            h5a.close(aid)

        self.assertRaises(TypeError, h5a.get_type, -1)

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

        self.assertRaises(TypeError, h5a.iterate, -1, iterate_two, namelist)


    # === Python extensions ===================================================

    def test_py_listattrs(self):
        self.assertEqual(h5a.py_listattrs(self.obj), ATTRIBUTES_ORDER)
        self.assertRaises(TypeError, h5a.py_listattrs, -1)

    def test_py_shape(self):
        
        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            aid = h5a.open_name(self.obj, name)
            retshape = h5a.py_shape(aid)
            self.assertEqual(retshape, shape) 
            h5a.close(aid)
        self.assertRaises(TypeError, h5a.py_shape, -1)

    def test_py_dtype(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            aid = h5a.open_name(self.obj, name)
            self.assertEqual(h5a.py_dtype(aid),dt)
            h5a.close(aid)
        self.assertRaises(TypeError, h5a.py_dtype, -1)

    def test_py_get(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            arr_reference = array(value, dtype=dt)
            arr_returned = h5a.py_get(self.obj, name)
            self.assert_(all(arr_returned == arr_reference), 
                errstr(arr_reference, arr_returned))
        self.assertRaises(TypeError, h5a.py_get, -1, "foo")

    def test_py_set(self):

        fid, filename = getcopy(HDFNAME)
        obj = h5g.open(fid, OBJECTNAME)

        for name, (value, dt, shape) in NEW_ATTRIBUTES.iteritems():
            arr_reference = array(value, dtype=dt)
            h5a.py_set(obj, name, arr_reference)
            arr_ret = h5a.py_get(obj, name)
            self.assert_( all( arr_ret == arr_reference), errstr(arr_ret, arr_reference))
        h5g.close(obj)
        deletecopy(fid, filename)

        self.assertRaises(TypeError, h5a.py_set, -1, "foo", arr_reference)


    def test_py_exists(self):

        for name in ATTRIBUTES:
            self.assert_(h5a.py_exists(self.obj, name), name)

        self.assert_(not h5a.py_exists(self.obj, 'SOME OTHER ATTRIBUTE') )
            
        # py_exists will never intentionally raise an exception











