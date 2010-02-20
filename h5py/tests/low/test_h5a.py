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

import numpy as np
from h5py import tests

from h5py import *

def isattr(aid):
    return aid and h5i.get_type(aid) == h5i.ATTR

class TestCreate(tests.HTest):

    def setUp(self):
        self.fid, self.name = tests.gettemp()
        self.sid = h5s.create_simple((1,))

    def tearDown(self):
        import os
        self.fid.close()
        os.unlink(self.name)

    def test_create_1(self):
        """ (H5A) Create attribute from type and space """
        aid = h5a.create(self.fid, 'name', h5t.STD_I32LE, self.sid)
        self.assert_(isattr(aid))
    
    @tests.require(api=18)
    def test_create_2(self):
        """ (H5A) Create attribute on group member w/custom LAPL """
        gid = h5g.create(self.fid, 'subgroup')
        lapl = h5p.create(h5p.LINK_ACCESS)
        aid = h5a.create(self.fid, 'name', h5t.STD_I32LE, self.sid,
                         obj_name='subgroup', lapl=lapl)
        self.assert_(isattr(aid))

    def test_exc_1(self):
        """ (H5A) Existing name causes ValueError """
        h5a.create(self.fid, 'name', h5t.STD_I32LE, self.sid)
        self.assertRaises(ValueError, h5a.create, self.fid, 'name',
                          h5t.STD_I32LE, self.sid)

    @tests.require(api=18)
    def test_exc_2(self):
        """ (H5A) Wrong obj_name causes KeyError """
        self.assertRaises(KeyError, h5a.create, self.fid, 'name',
                          h5t.STD_I32LE, self.sid, obj_name='missing')

class TestOpenExists(tests.HTest):

    def setUp(self):
        self.fid, self.name = tests.gettemp()
        tid = h5t.STD_I32LE
        sid = h5s.create_simple((1,))
        h5a.create(self.fid, 'name', tid, sid)
        gid = h5g.create(self.fid, 'subgroup')
        h5a.create(gid, 'othername', tid, sid)

    def tearDown(self):
        self.fid.close()
        import os
        os.unlink(self.name)

    def test_open_1(self):
        """ (H5A) Open by name """
        aid = h5a.open(self.fid, 'name')
        self.assert_(isattr(aid))

    def test_open_2(self):
        """ (H5A) Open by index """
        aid = h5a.open(self.fid, index=0)
        self.assert_(isattr(aid))

    @tests.require(api=18)
    def test_open_3(self):
        """ (H5A) Open from group member w/custom LAPL """
        lapl = h5p.create(h5p.LINK_ACCESS)
        aid = h5a.open(self.fid, 'othername', obj_name='subgroup', lapl=lapl)
        self.assert_(isattr(aid))
    
    def test_exists_1(self):
        """ (H5A) Check exists by name """
        self.assert_(h5a.exists(self.fid, 'name') is True)
        self.assert_(h5a.exists(self.fid, 'missing') is False)

    @tests.require(api=18)
    def test_exists_2(self):
        """ (H5A) Check exists on group member with custom LAPL """
        lapl = h5p.create(h5p.LINK_ACCESS)
        self.assert_(h5a.exists(self.fid, 'othername', obj_name='subgroup', lapl=lapl) is True)
        self.assert_(h5a.exists(self.fid, 'missing', obj_name='subgroup', lapl=lapl) is False)

    def test_exc_1(self):
        """ (H5A) Open with wrong name causes KeyError """
        self.assertRaises(KeyError, h5a.open, self.fid, 'missing')

    #def test_exc_2(self):
    #    """ (H5A) Open with wrong index causes ValueError """
    #    self.assertRaises(ValueError, h5a.open, self.fid, index=2)

    @tests.require(api=18)
    def test_exc_3(self):
        """ (H5A) Open with wrong subgroup causes KeyError """
        self.assertRaises(KeyError, h5a.open, self.fid, 'othername', obj_name='missing')


