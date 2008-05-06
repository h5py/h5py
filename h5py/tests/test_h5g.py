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

import h5py
from h5py import h5f, h5g, h5i
from h5py.errors import GroupError

from common import getcopy, deletecopy

HDFNAME = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/attributes.hdf5')
OBJECTNAME = 'Group'
TEST_GROUPS = ['Subgroup1','Subgroup2','Subgroup3']
NEW_LINK_NAME = 'Link name'

class TestH5G(unittest.TestCase):

    def setUp(self):
        self.fid = h5f.open(HDFNAME, h5f.ACC_RDONLY)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        h5g.close(self.obj)
        h5f.close(self.fid)

    def is_grp(self, item):
        return h5i.get_type(item) == h5i.TYPE_GROUP

    def test_open_close(self):
        for name in TEST_GROUPS:
            gid = h5g.open(self.obj, name)
            self.assert_(self.is_grp(gid))
            h5g.close(gid)
            self.assert_(not self.is_grp(gid))
        
        self.assertRaises(GroupError, h5g.open, self.obj, 'Some other group')
        self.assertRaises(GroupError, h5g.close, -1)

    def test_create(self):
        fid, filename = getcopy(HDFNAME)
        obj = h5g.open(fid, OBJECTNAME)

        gid = h5g.create(obj, 'New group')
        h5g.close(gid)
        self.assert_(h5g.py_exists(obj, 'New group'))
        self.assertRaises(GroupError, h5g.create, obj, 'New group')

        deletecopy(fid, filename)

    def test_link_unlink_move(self):
        fid, filename = getcopy(HDFNAME)
        obj = h5g.open(fid, OBJECTNAME)

        # local link
        h5g.link(obj, TEST_GROUPS[1], NEW_LINK_NAME, h5g.LINK_HARD)
        self.assert_( h5g.py_exists(obj, NEW_LINK_NAME) )

        # test local unlink
        h5g.unlink(obj, NEW_LINK_NAME)
        self.assert_(not h5g.py_exists(obj, NEW_LINK_NAME))

        # remote link
        rgid = h5g.open(obj, TEST_GROUPS[0])
        h5g.link(obj, TEST_GROUPS[0], NEW_LINK_NAME, h5g.LINK_HARD, rgid)
        self.assert_( h5g.py_exists(rgid, NEW_LINK_NAME) )
    
        h5g.unlink(rgid, NEW_LINK_NAME)
        self.assert_( not h5g.py_exists(rgid, NEW_LINK_NAME) )
        h5g.close(rgid)

        h5g.move(obj, TEST_GROUPS[2], NEW_LINK_NAME)
        self.assert_(h5g.py_exists(obj, NEW_LINK_NAME))
        self.assert_(not h5g.py_exists(obj, TEST_GROUPS[2]))

        self.assertRaises(GroupError, h5g.move, obj, 'Ghost group', 'blah')
        self.assertRaises(GroupError, h5g.unlink, obj, 'Some other name')
        self.assertRaises(GroupError, h5g.link, obj, 'Ghost group', 'blah') 

        h5g.close(obj)

        deletecopy(fid, filename)

    def test_get_num_objs(self):

        self.assertEqual(h5g.get_num_objs(self.obj), 3)
        self.assertRaises(GroupError, h5g.get_num_objs, -1)

    def test_objname_objtype(self):

        for idx, name in enumerate(TEST_GROUPS):
            self.assertEqual(h5g.get_objname_by_idx(self.obj, idx), name)
            self.assertEqual(h5g.get_objtype_by_idx(self.obj, idx), h5g.OBJ_GROUP)

        self.assertRaises(GroupError, h5g.get_objname_by_idx, self.obj, -1)
        self.assertRaises(GroupError, h5g.get_objtype_by_idx, self.obj, -1)

    def test_get_objinfo(self):

        retval = h5g.get_objinfo(self.obj, '.')
        retval.fileno
        retval.objno
        self.assertEqual(retval.nlink, 1)
        self.assertEqual(retval.type, h5g.OBJ_GROUP)
        retval.mtime
        retval.linklen

        self.assertRaises(GroupError, h5g.get_objinfo, self.obj, 'Something else')


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
        n = h5g.iterate(self.obj, '.', iterate_all, namelist)
        self.assertEqual(namelist, TEST_GROUPS)
        self.assertEqual(n, len(TEST_GROUPS)-1)

        namelist = []
        n = h5g.iterate(self.obj, '.', iterate_two, namelist)
        self.assertEqual(namelist, TEST_GROUPS[0:2])
        self.assertEqual(n, 1)

        namelist = []
        self.assertRaises(RuntimeError, h5g.iterate, self.obj, '.', iterate_fault, namelist)
        self.assertEqual(namelist, TEST_GROUPS[0:2])
        
        namelist = []
        n = h5g.iterate(self.obj, '.', iterate_two, namelist, 1)
        self.assertEqual(namelist, TEST_GROUPS[1:3])
        self.assertEqual(n, 2)

    def test_py_listnames(self):

        self.assertEqual(h5g.py_listnames(self.obj), TEST_GROUPS)
        self.assertRaises(GroupError, h5g.py_listnames, -1)

    def test_py_iternames(self):

        iterator = h5g.py_iternames(self.obj)
        self.assertEqual(list(iterator), TEST_GROUPS)
        #self.assertRaises(StopIteration, iterator.next()) bug in unittest
        
        self.assertRaises(GroupError, h5g.py_iternames, -1)

    def test_py_exists(self):

        self.assert_(h5g.py_exists(self.obj, TEST_GROUPS[0]))
        self.assert_(not h5g.py_exists(self.obj, 'Something else'))



    






























