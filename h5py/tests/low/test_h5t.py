
"""
    H5T tests.  This handles the general API behavior and exceptions; full
    type-specific tests (including conversion) are tested elsewhere.
"""

from h5py import tests
from h5py import *

import numpy as np

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

    def test_equal(self):
        """ (H5T) equal() """
        tid = h5t.STD_I32LE.copy()
        self.assert_(tid.equal(h5t.STD_I32LE))
        self.assert_(h5t.STD_I32LE.equal(tid))
        self.assert_(not tid.equal(h5t.STD_I32BE))

    def test_get_class(self):
        """ (H5T) get_class() """
        self.assertEqual(h5t.STD_I32LE.get_class(), h5t.INTEGER)

class TestEncodeDecode(Base):

    def setUp(self):
        self.tid = h5t.STD_I32LE.copy()

    def tearDown(self):
        del self.tid

    @tests.require(api=18)
    def test_ed(self):
        """ (H5T) Encode/decode round trip """
        enc = self.tid.encode()
        self.assertIsInstance(enc, str)
        dec = h5t.decode(enc)
        self.assertEqual(self.tid, dec)

    @tests.require(api=18)
    def test_pickle(self):
        """ (H5T) Encode/decode round trip via pickling """
        import pickle
        pkl = pickle.dumps(self.tid)
        dec = pickle.loads(pkl)
        self.assertEqual(self.tid, dec)

class TestFloat(Base):

    @tests.require(hasattr(np, 'float128'))
    def test_float_exc(self):
        """ (H5T) Unsupported float size raises TypeError """
        self.assertRaises(TypeError, h5t.py_create, np.float128)

class TestInteger(Base):

    def test_order(self):
        """ (H5T) integer byte order """
        tid = h5t.STD_I32LE.copy()
        self.assertEqual(tid.get_order(), h5t.ORDER_LE)
        tid.set_order(h5t.ORDER_BE)
        self.assertEqual(tid.get_order(), h5t.ORDER_BE)

    def test_sign(self):
        """ (H5T) integer sign """
        tid = h5t.STD_I32LE.copy()
        self.assertEqual(tid.get_sign(), h5t.SGN_2)
        tid.set_sign(h5t.SGN_NONE)
        self.assertEqual(tid.get_sign(), h5t.SGN_NONE)

    
