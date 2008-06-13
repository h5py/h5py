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

import h5py
from h5py import h5s, h5i
from h5py.h5e import H5Error

spaces = [(10,10), (1,1), (1,), ()]
max_spaces = [(10,10), (3,4), (h5s.UNLIMITED,), ()]

class TestH5S(unittest.TestCase):


    def test_create_close(self):
        sid = h5s.create(h5s.SCALAR)
        self.assertEqual(h5i.get_type(sid), h5i.DATASPACE)
        h5s.close(sid)
        self.assertEqual(h5i.get_type(sid), h5i.BADID)

        self.assertRaises(H5Error, h5s.create, -1)
        self.assertRaises(H5Error, h5s.close, -1)

    def test_copy(self):
        sid = h5s.create(h5s.SCALAR)
        sid2 = h5s.copy(sid)
        self.assertEqual(h5i.get_type(sid2), h5i.DATASPACE)
        h5s.close(sid2)
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.copy, -1)

    def test_simple(self):
        # Tests create_simple, get_simple_extent_dims, get_simple_extent_ndims

        for space, max_space in zip(spaces, max_spaces):
            sid = h5s.create_simple(space,max_space)
            self.assertEqual(h5s.get_simple_extent_dims(sid), space)
            self.assertEqual(h5s.get_simple_extent_dims(sid, True), max_space)
            self.assertEqual(h5s.get_simple_extent_ndims(sid), len(space))
            h5s.close(sid)

        # Bad input
        self.assertRaises(ValueError, h5s.create_simple, None)
        self.assertRaises(H5Error, h5s.get_simple_extent_dims, -1)
        self.assertRaises(H5Error, h5s.get_simple_extent_ndims, -1)

        # Illegal input
        self.assertRaises(H5Error, h5s.create_simple, (10,10), (10,9))
        self.assertRaises(ValueError, h5s.create_simple, (10,10), (10,))

    def test_is_simple(self):
        # According to HDF5 docs, all dataspaces are "simple," even scalar ones.
        sid = h5s.create(h5s.SCALAR)
        self.assert_(h5s.is_simple(sid))
        h5s.close(sid)
        sid = h5s.create(h5s.SIMPLE)    
        self.assert_(h5s.is_simple(sid))
        h5s.close(sid)

        # I think this should be H5Error but the library disagrees.
        self.assertRaises(H5Error, h5s.is_simple, -1)

    def test_offset_simple(self):
        
        sid = h5s.create_simple((100,100))
        h5s.select_hyperslab(sid, (0,0), (10,10))
        self.assertEqual(h5s.get_select_bounds(sid), ((0,0),(9,9)))
        h5s.offset_simple(sid,(2,2))
        self.assertEqual(h5s.get_select_bounds(sid), ((2,2),(11,11)))
        h5s.offset_simple(sid, None)
        self.assertEqual(h5s.get_select_bounds(sid), ((0,0),(9,9)))

        self.assertRaises(H5Error, h5s.offset_simple, -1, (10,10))
        self.assertRaises(ValueError, h5s.offset_simple, sid, (10,))

        h5s.close(sid)

    def test_get_simple_extent_npoints(self):
        
        sid = h5s.create_simple((100,100))
        self.assertEqual(h5s.get_simple_extent_npoints(sid), 100*100)
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.get_simple_extent_npoints, -1)

    def test_get_simple_extent_type(self):

        sid = h5s.create(h5s.SIMPLE)
        sid2 = h5s.create(h5s.SCALAR)
        self.assertEqual(h5s.get_simple_extent_type(sid), h5s.SIMPLE)
        self.assertEqual(h5s.get_simple_extent_type(sid2), h5s.SCALAR)
        h5s.close(sid)
        h5s.close(sid2)

        self.assertRaises(H5Error, h5s.get_simple_extent_type, -1)

    def test_extent_copy(self):

        sid = h5s.create(h5s.SIMPLE)
        sid2 = h5s.create_simple((35,42))
        h5s.extent_copy(sid, sid2)
        self.assertEqual(h5s.get_simple_extent_dims(sid2), (35,42))
        h5s.close(sid)
        h5s.close(sid2)

        self.assertRaises(H5Error, h5s.extent_copy, -1, -1)

    def test_set_extent_simple(self):

        for space, max_space in zip(spaces, max_spaces):
            sid = h5s.create_simple((10,10))
            h5s.set_extent_simple(sid, space, max_space)
            self.assertEqual(h5s.get_simple_extent_dims(sid), space)
            self.assertEqual(h5s.get_simple_extent_dims(sid, True), max_space)
            h5s.close(sid)

        self.assertRaises(H5Error, h5s.set_extent_simple, -1, (10,10))

    def test_set_extent_none(self):

        sid = h5s.create_simple((10,10))
        self.assertEqual(h5s.get_simple_extent_type(sid), h5s.SIMPLE)
        h5s.set_extent_none(sid)
        self.assertEqual(h5s.get_simple_extent_type(sid), h5s.NO_CLASS)
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.set_extent_none, -1)


    def test_get_select_type_npoints(self):

        sid = h5s.create_simple((10,10))
        h5s.select_hyperslab(sid, (0,0), (5,5))
        self.assertEqual(h5s.get_select_type(sid), h5s.SEL_HYPERSLABS)
        self.assertEqual(h5s.get_select_npoints(sid), 25)
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.get_select_type, -1)
        self.assertRaises(H5Error, h5s.get_select_npoints, -1)

    def test_get_select_bounds(self):

        sid = h5s.create_simple((100,100))
        h5s.select_all(sid)
        self.assertEqual(h5s.get_select_bounds(sid), ((0,0), (99,99)))
        h5s.select_hyperslab(sid, (10,10), (13,17))
        self.assertEqual(h5s.get_select_bounds(sid), ((10,10), (22,26)))
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.get_select_bounds, -1)

    def test_select(self):

        sid = h5s.create_simple((100,100))
        h5s.select_none(sid)
        self.assertEqual(h5s.get_select_npoints(sid), 0)
        h5s.select_all(sid)
        self.assertEqual(h5s.get_select_npoints(sid), 100*100)
        h5s.select_none(sid)
        self.assertEqual(h5s.get_select_npoints(sid), 0)

        h5s.select_hyperslab(sid, (0,0), (10,10))
        self.assert_(h5s.select_valid(sid))
        h5s.select_hyperslab(sid, (0,0), (200,200))
        self.assert_(not h5s.select_valid(sid))
        
        h5s.close(sid)

        self.assertRaises(H5Error, h5s.select_none, -1)
        self.assertRaises(H5Error, h5s.select_all, -1)
        self.assertRaises(H5Error, h5s.select_valid, -1)

    def test_elements(self):

        pointlist= [(0,0), (15,98), (4,17), (67,32)] 
        sid = h5s.create_simple((100,100))

        h5s.select_elements(sid, pointlist)
        self.assertEqual(h5s.get_select_elem_npoints(sid), len(pointlist))
        self.assertEqual(h5s.get_select_elem_pointlist(sid), pointlist)

        self.assertRaises(H5Error, h5s.select_elements, -1, [])
        self.assertRaises(H5Error, h5s.get_select_elem_npoints, -1)
        self.assertRaises(H5Error, h5s.get_select_elem_pointlist, -1)

    def test_get_blocks(self):

        start = [ (0,0), (50,60) ]
        count = [ (5,5), (13,17) ]

        sid = h5s.create_simple((100,100))
        h5s.select_hyperslab(sid, start[0], count[0], op=h5s.SELECT_SET)
        h5s.select_hyperslab(sid, start[1], count[1], op=h5s.SELECT_OR)

        self.assertEqual(h5s.get_select_hyper_nblocks(sid), 2)
        blocklist = h5s.get_select_hyper_blocklist(sid)
        self.assertEqual(blocklist, [( (0,0), (4,4) ), ( (50,60), (62,76) )])

        h5s.close(sid)

        self.assertRaises(H5Error, h5s.get_select_hyper_nblocks, -1)
        self.assertRaises(H5Error, h5s.get_select_hyper_blocklist, -1)



















