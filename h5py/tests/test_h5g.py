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

from common import TestCasePlus

from h5py import *

HDFNAME = 'attributes.hdf5'
OBJECTNAME = 'Group'
TEST_GROUPS = ['Subgroup1','Subgroup2','Subgroup3']
NEW_LINK_NAME = 'Link name'

class TestH5G(TestCasePlus):


    def setUp(self):
        self.setup_fid(HDFNAME)
        self.obj = h5g.open(self.fid, OBJECTNAME)

    def tearDown(self):
        self.obj._close()
        self.teardown_fid()

    def is_grp(self, item):
        return h5i.get_type(item) == h5i.GROUP

    def test_open_close(self):
        for name in TEST_GROUPS:
            grp = h5g.open(self.obj, name)
            self.assert_(self.is_grp(grp))
            pr = str(grp)
            grp._close()
            self.assert_(not self.is_grp(grp), pr+"::"+str(grp))
        
        self.assertRaises(KeyError, h5g.open, self.obj, 'Some other group')

    def test_create(self):

        obj = h5g.open(self.fid, OBJECTNAME)
        grp = h5g.create(obj, 'New group')
        grp._close()
        self.assert_('New group' in obj)

    def test_link_A(self):

        obj = h5g.open(self.fid, OBJECTNAME)

        # symlink
        obj.link(TEST_GROUPS[1], NEW_LINK_NAME, h5g.LINK_SOFT)
        self.assertEqual(h5g.get_objinfo(obj, NEW_LINK_NAME, follow_link=False).type, h5g.LINK)
        self.assertEqual(obj.get_linkval(NEW_LINK_NAME), TEST_GROUPS[1])

    def test_link_B(self):

        obj = h5g.open(self.fid, OBJECTNAME)

        # local link
        obj.link(TEST_GROUPS[1], NEW_LINK_NAME, h5g.LINK_HARD)
        self.assert_( NEW_LINK_NAME in obj )

        # test local unlink
        obj.unlink(NEW_LINK_NAME)
        self.assert_(not NEW_LINK_NAME in obj)

        # remote link
        rgrp = h5g.open(obj, TEST_GROUPS[0])
        obj.link(TEST_GROUPS[0], NEW_LINK_NAME, h5g.LINK_HARD, rgrp)
        self.assert_( NEW_LINK_NAME in rgrp )
    
        # remote unlink
        rgrp.unlink(NEW_LINK_NAME)
        self.assert_( not NEW_LINK_NAME in rgrp )

        # move
        obj.move( TEST_GROUPS[2], NEW_LINK_NAME)
        self.assert_(NEW_LINK_NAME in obj)
        self.assert_(not TEST_GROUPS[2] in obj)


    def test_get_num_objs(self):

        self.assertEqual(self.obj.get_num_objs(), 3)


    def test_objname_objtype(self):

        for idx, name in enumerate(TEST_GROUPS):
            self.assertEqual(self.obj.get_objname_by_idx(idx), name)
            self.assertEqual(self.obj.get_objtype_by_idx(idx), h5g.GROUP)


    def test_get_objinfo(self):

        retval = h5g.get_objinfo(self.obj)
        retval.fileno
        retval.objno
        self.assertEqual(retval.nlink, 1)
        self.assertEqual(retval.type, h5g.GROUP)
        retval.mtime
        retval.linklen

    def test_iterate(self):

        namelist = []
        h5g.iterate(self.obj, namelist.append)
        self.assertEqual(namelist, TEST_GROUPS)

        namelist = []
        def iterate_two(name):
            if len(namelist) == 2:
                return True
            namelist.append(name)
        result = h5g.iterate(self.obj, iterate_two)
        self.assertEqual(namelist, TEST_GROUPS[0:2])
        self.assert_(result is True)

        def iter_fault(name):
            raise RuntimeError
        self.assertRaises(RuntimeError, h5g.iterate, self.obj, iter_fault)

    def test_get_set_comment(self):

        obj = h5g.open(self.fid, OBJECTNAME)

        obj.set_comment(TEST_GROUPS[0], "This is a comment.")
        self.assertEqual(obj.get_comment(TEST_GROUPS[0]), "This is a comment.")


    def test_py_contains(self):

        self.assert_(TEST_GROUPS[0] in self.obj)
        self.assert_(not 'Something else' in self.obj)

    def test_py_iter(self):
        
        namelist = list(self.obj)
        self.assertEqual(namelist, TEST_GROUPS)

    






























