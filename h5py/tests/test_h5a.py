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
from numpy import array, ndarray, dtype, all, ones
import os

from common import HCopy, errstr

import h5py
from h5py import h5, h5a, h5f, h5g, h5i, h5t, h5s
from h5py.h5 import H5Error


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
        self.obj.close()
        self.fid.close()

    def is_attr(self, attr):
        return (h5i.get_type(attr) == h5i.ATTR)

    # === General attribute operations ========================================

    def test_create_write(self):

        with HCopy(HDFNAME) as fid:
            obj = h5g.open(fid, OBJECTNAME)
            for name, (value, dt, shape) in NEW_ATTRIBUTES.iteritems():
                arr_ref = array(value, dtype=dt)
                arr_fail = ones((15,15), dtype=dt)

                space = h5s.create(h5s.SCALAR)
                htype = h5t.py_create(dt)

                attr = h5a.create(obj, name, htype, space)
                self.assert_(self.is_attr(attr))
                attr.write(arr_ref)
                self.assertRaises(ValueError, attr.write, arr_fail)
                attr.close()

                attr = h5a.open_name(obj, name)
                dt = attr.dtype
                shape = attr.shape
                arr_val = ndarray(shape, dtype=dt)
                attr.read(arr_val)
                attr.close()
                self.assert_(all(arr_val == arr_ref), errstr(arr_val, arr_ref))

            obj.close()

    def test_open_idx(self):
        for idx, name in enumerate(ATTRIBUTES_ORDER):
            attr = h5a.open_idx(self.obj, idx)
            self.assert_(self.is_attr(attr), "Open: index %d" % idx)
            attr.close()

    def test_open_name(self):
        for name in ATTRIBUTES:
            attr = h5a.open_name(self.obj, name)
            self.assert_(self.is_attr(attr), 'Open: name "%s"' % name)
            attr.close()

    def test_close(self):
        attr = h5a.open_idx(self.obj, 0)
        self.assert_(self.is_attr(attr))
        attr.close()
        self.assert_(not self.is_attr(attr))

    def test_delete(self):
        with HCopy(HDFNAME) as fid:
            obj = h5g.open(fid, OBJECTNAME)

            attr = h5a.open_name(obj, ATTRIBUTES_ORDER[0])
            self.assert_(self.is_attr(attr))
            attr.close()

            h5a.delete(obj, ATTRIBUTES_ORDER[0])
            self.assertRaises(H5Error, h5a.open_name, obj, ATTRIBUTES_ORDER[0])


    # === Attribute I/O =======================================================

    def test_read(self):
        for name in ATTRIBUTES:
            value, dt, shape = ATTRIBUTES[name]

            attr = h5a.open_name(self.obj, name)
            arr_holder = ndarray(shape, dtype=dt)
            arr_reference = array(value, dtype=dt)

            self.assertEqual(attr.shape, shape)
            self.assertEqual(attr.dtype, dt)

            attr.read(arr_holder)
            self.assert_( all(arr_holder == arr_reference),
                errstr(arr_reference, arr_holder, 'Attr "%s"):\n' % name, ))

            attr.close()

    # write is done by test_create_write

    # === Attribute inspection ================================================

    def test_get_num_attrs(self):
        n = h5a.get_num_attrs(self.obj)
        self.assertEqual(n, len(ATTRIBUTES))

    def test_get_name(self):
    
        for name in ATTRIBUTES:
            attr = h5a.open_name(self.obj, name)
            self.assertEqual(attr.get_name(), name)

    def test_get_space(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open_name(self.obj, name)
            space = attr.get_space()
            shape_tpl = space.get_simple_extent_dims()
            self.assertEqual(shape_tpl, shape)

    def test_get_type(self):

        for name, (value, dt, shape) in ATTRIBUTES.iteritems():
            attr = h5a.open_name(self.obj, name)
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












