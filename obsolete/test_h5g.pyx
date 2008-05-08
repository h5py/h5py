##### Preamble block ##########################################################
# 
# This file is part of the "h5py" HDF5 Interface for Python project.
# 
# Copyright 2008 Andrew Collette
# http://software.alfven.org
# License: BSD  (See file "LICENSE" for complete license, or the URL above)
# 
##### End preamble block ######################################################

from defs_h5g cimport H5Gclose, H5Gopen, H5Gget_objinfo, H5Gunlink, \
                      H5G_GROUP, H5G_stat_t
from defs_h5i cimport H5Iget_type, H5I_GROUP, H5I_BADID

import unittest
import shutil
import tempfile
import os

import h5f
import h5g

from errors import GroupError

""" Depends on h5f
"""

TEST_FILE = "test_simple.hdf5"
TEST_GROUPS = ["columns", "detector"]
NEW_GROUP_NAME = "XXXNEWGROUPXXX"
NEW_LINK_NAME = "linked"

# Pyrex doesn't let you nest functions
# These are used in the iterator test.
def ifunc1(gid, name, data):
    data.append(name)
    return None

def ifunc2(gid, name, data):
    data.append(name)
    return 0

def ifunc3(gid, name, data):
    data.append(name)
    return 1

def ifunc4(gid, name, data):
    data.append(name)
    return -1


class TestH5G(unittest.TestCase):

    def setUp(self):
        self.fname = tempfile.mktemp(".hdf5")
        shutil.copyfile(TEST_FILE, self.fname)
        self.fid = h5f.open(self.fname, flags=h5f.ACC_RDWR)

    def tearDown(self):
        h5f.close(self.fid)
        os.unlink(self.fname)

    def testopen(self):
        for name in TEST_GROUPS:
            gid = h5g.open(self.fid, name)
            self.assert_(H5Iget_type(gid) == H5I_GROUP)
            H5Gclose(gid)

    def testclose(self):
        for name in TEST_GROUPS:
            gid = H5Gopen(self.fid, name)
            h5g.close(gid)
            self.assert_(H5Iget_type(gid) == H5I_BADID)

    def testcreate(self):
        gid = h5g.create(self.fid, NEW_GROUP_NAME)
        self.assert_( H5Gget_objinfo(self.fid, NEW_GROUP_NAME, 0, NULL) >= 0 )
        H5Gclose(gid)
        H5Gunlink(self.fid, NEW_GROUP_NAME)

    def testlink(self):
        # local link
        h5g.link(self.fid, TEST_GROUPS[1], NEW_LINK_NAME, h5g.LINK_HARD)
        self.assert_( H5Gget_objinfo(self.fid, NEW_LINK_NAME, 0, NULL) >= 0 )

        # test local unlink
        h5g.unlink(self.fid, NEW_LINK_NAME)
        self.assert_( H5Gget_objinfo(self.fid, NEW_LINK_NAME, 0, NULL) < 0 )

        # remote link
        rgid = H5Gopen(self.fid, TEST_GROUPS[0])
        h5g.link(self.fid, TEST_GROUPS[0], NEW_LINK_NAME, h5g.LINK_HARD, rgid)
        self.assert_( H5Gget_objinfo(rgid, NEW_LINK_NAME, 0, NULL) >= 0 )
    
        h5g.unlink(rgid, NEW_LINK_NAME)
        self.assert_( H5Gget_objinfo(rgid, NEW_LINK_NAME, 0, NULL) < 0 )

    def testmove(self):
        tname = TEST_GROUPS[0]+'_2'

        # local move
        h5g.move(self.fid, TEST_GROUPS[0], tname)
        self.assert_( H5Gget_objinfo(self.fid, TEST_GROUPS[0], 0, NULL) < 0 )        
        self.assert_( H5Gget_objinfo(self.fid, tname, 0, NULL) >= 0 )

        h5g.move(self.fid, TEST_GROUPS[0]+'_2', TEST_GROUPS[0])
        self.assert_( H5Gget_objinfo(self.fid, TEST_GROUPS[0], 0, NULL) >= 0 )        
        self.assert_( H5Gget_objinfo(self.fid, tname, 0, NULL) <0 )

        gid = H5Gopen(self.fid, TEST_GROUPS[1])

        # remote move
        h5g.move(self.fid, TEST_GROUPS[0], TEST_GROUPS[0], gid)
        self.assert_( H5Gget_objinfo(self.fid, TEST_GROUPS[0], 0, NULL) < 0 )    
        self.assert_( H5Gget_objinfo(gid, TEST_GROUPS[0], 0, NULL) >= 0 )

        h5g.move(gid, TEST_GROUPS[0], TEST_GROUPS[0], self.fid)
        self.assert_( H5Gget_objinfo(self.fid, TEST_GROUPS[0], 0, NULL) >= 0 )    
        self.assert_( H5Gget_objinfo(gid, TEST_GROUPS[0], 0, NULL) < 0 )

        H5Gclose(gid)

    def test_get_num_objs(self):
        self.assert_(h5g.get_num_objs(self.fid) == 2)

    def test_get_objname_by_idx(self):

        for idx, name in enumerate(TEST_GROUPS):
            self.assert_(h5g.get_objname_by_idx(self.fid, idx) == name)

    def test_get_objtype_by_idx(self):

        for idx, name in enumerate(TEST_GROUPS):
            self.assert_(h5g.get_objtype_by_idx(self.fid, idx) == h5g.OBJ_GROUP)
        
    def test_get_objinfo(self):

        cdef H5G_stat_t stat
        H5Gget_objinfo(self.fid, TEST_GROUPS[0], 1, &stat)
        qstat = h5g.get_objinfo(self.fid, TEST_GROUPS[0])

        self.assert_(qstat.fileno[0] == stat.fileno[0])
        self.assert_(qstat.fileno[1] == stat.fileno[1])
        self.assert_(qstat.nlink == stat.nlink)
        self.assert_(qstat.type == <int>stat.type)
        self.assert_(qstat.mtime == stat.mtime)
        self.assert_(qstat.linklen == stat.linklen)

    def test_iterate(self):

        nlist = []
        h5g.iterate(self.fid, '.', ifunc1, nlist)
        self.assert_(nlist == TEST_GROUPS )
        
        nlist = []
        h5g.iterate(self.fid, '.', ifunc2, nlist)
        self.assert_(nlist == TEST_GROUPS)

        nlist = []
        h5g.iterate(self.fid, '.', ifunc3, nlist)
        self.assert_(nlist == [TEST_GROUPS[0]])

        nlist = []
        self.assertRaises(GroupError, h5g.iterate, self.fid, '.', ifunc4, nlist)
        self.assert_(nlist == [TEST_GROUPS[0]])

    def test_py_listnames(self):

        thelist = h5g.py_listnames(self.fid)
        self.assert_(thelist == TEST_GROUPS)

        
    def test_py_iternames(self):
        iterator = h5g.py_iternames(self.fid)
        thelist = list( iterator )
        self.assert_(thelist == TEST_GROUPS)
        self.assertRaises(StopIteration, iterator.next)
        




    

