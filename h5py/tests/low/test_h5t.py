
"""
    H5T tests.  This handles the general API behavior and exceptions; full
    type-specific tests (including conversion) are tested elsewhere.
"""

from h5py import tests
from h5py import *

class Base(tests.HTest):
    pass

class TestCreate(Base):

    def test_create(self):
        """ (H5T) create() """
        tid = h5t.create(h5t.OPAQUE, 72)
        self.assertIsInstance(tid, h5t.TypeOpaqueID)

    def test_exc(self):
        """ (H5T) ValueError for non-opaque, non-compound class """
        self.assertRaises(ValueError, h5t.create, h5t.INTEGER, 4)

class TestCommit(Base):

    def setUp(self):
        self.fid, self.name = tests.gettemp()

    def tearDown(self):
        import os
        self.fid.close()
        os.unlink(self.name)

    def test_commit_committed(self):
        """ (H5T) Commit type changes committed() """
        tid = h5t.STD_I32LE.copy()
        self.assert_(not tid.committed())
        tid.commit(self.fid, 'name')
        self.assert_(tid.committed())

    @tests.require(api=18)
    def test_commit_pl(self):
        """ (H5T) Commit type with non-default LCPL """
        tid = h5t.STD_I32LE.copy()
        tid.commit(self.fid, 'name', lcpl=h5p.create(h5p.LINK_CREATE))
        self.assert_(tid.committed())

    def test_open(self):
        """ (H5T) Open committed type """
        tid = h5t.STD_I32LE.copy()
        tid.commit(self.fid, 'name')
        tid2 = h5t.open(self.fid, 'name')
        self.assertEqual(tid, tid2)

    def test_open_exc(self):
        """ (H5T) Open missing type raises KeyError """
        self.assertRaises(KeyError, h5t.open, self.fid, 'missing')

class TestTypeID(Base):

    """ Common simple TypeID operations """

    def test_copy(self):
        """ (H5T) copy() """
        tid = h5t.create(h5t.OPAQUE, 72)
        tid2 = tid.copy()
        self.assertEqual(tid, tid2)
        self.assert_(tid is not tid2)
     
    def test_lock(self):
        """ (H5T) Modification of locked type raises TypeError """
        htype = h5t.STD_I8LE.copy()
        htype.set_sign(h5t.SGN_NONE)
        htype.lock()
        self.assertRaises(TypeError, htype.set_sign, h5t.SGN_2)

    def test_equal(self):
        """ (H5T) equal() """
        tid = h5t.STD_I32LE.copy()
        self.assert_(tid.equal(h5t.STD_I32LE))
        self.assert_(h5t.STD_I32LE.equal(tid))
        self.assert_(not tid.equal(h5t.STD_I32BE))

    def test_get_class(self):
        """ (H5T) get_class() """
        self.assertEqual(h5t.STD_I32LE.get_class(), h5t.INTEGER)

class TestFloat(Base):

    def test_float_exc(self):
        """ (H5T) Unsupported float size raises TypeError """
        import numpy
        if hasattr(numpy, 'float128'):
            self.assertRaises(TypeError, h5t.py_create, numpy.float128)

    

    
