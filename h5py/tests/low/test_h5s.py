
from h5py import tests
from h5py import *

class Base(tests.HTest):

    def assertEqualSpaces(self, sid1, sid2):
        self.assertIsInstance(sid1, h5s.SpaceID)
        self.assertIsInstance(sid2, h5s.SpaceID)
        self.assertEqual(sid1.shape, sid2.shape)
        self.assertEqual(sid1.get_select_bounds(), sid2.get_select_bounds())

class TestCreate(Base):

    def test_scalar(self):
        """ (H5S) Create scalar dataspace """
        sid = h5s.create(h5s.SCALAR)
        self.assertEqual(sid.get_simple_extent_type(), h5s.SCALAR)

    def test_simple(self):
        """ (H5S) Create simple dataspaces """
        sid = h5s.create(h5s.SIMPLE)
        self.assertEqual(sid.get_simple_extent_type(), h5s.SIMPLE)

    def test_simple_scalar(self):
        """ (H5S) Create from empty tuple results in scalar space """
        sid = h5s.create_simple(())
        self.assertEqual(sid.get_simple_extent_type(), h5s.SCALAR)

    def test_simple_init(self):
        """ (H5S) Create simple dataspace given extent """
        sid = h5s.create_simple((10,10))
        self.assertEqual(sid.shape, (10,10))
        self.assertEqual(sid.get_simple_extent_dims(maxdims=True), (10,10))

    def test_simple_limit(self):
        """ (H5S) Create simple dataspace given extent and limit """
        sid = h5s.create_simple((10,10), (15,15))
        self.assertEqual(sid.shape, (10,10))
        self.assertEqual(sid.get_simple_extent_dims(maxdims=True), (15,15))

    def test_simple_ulimit(self):
        """ (H5S) Create simple dataspace with unlimited dimension """
        sid = h5s.create_simple((10,10), (h5s.UNLIMITED,15))
        self.assertEqual(sid.shape, (10,10))
        self.assertEqual(sid.get_simple_extent_dims(maxdims=True), (h5s.UNLIMITED,15))

    def test_simple_exc(self):
        """ (H5S) Extent/limit mismatch raises ValueError """
        self.assertRaises(ValueError, h5s.create_simple, (10,10), (10,9))

    def test_simple_exc1(self):
        """ (H5S) Extent/limit rank mismatch raises ValueError """
        self.assertRaises(ValueError, h5s.create_simple, (10,10), (20,))

class TestEncodeDecode(Base):

    def setUp(self):
        self.sid = h5s.create_simple((10,10))
        self.sid.select_hyperslab((2,2),(5,5))

    def tearDown(self):
        del self.sid

    @tests.require(api=18)
    def test_ed(self):
        """ (H5S) Encode/decode round trip """
        enc = self.sid.encode()
        self.assertIsInstance(enc, str)
        dec = h5s.decode(enc)
        self.assertEqualSpaces(self.sid, dec)

    @tests.require(api=18)
    def test_pickle(self):
        """ (H5S) Encode/decode round trip via pickling """
        import pickle
        pkl = pickle.dumps(self.sid)
        dec = pickle.loads(pkl)
        self.assertEqualSpaces(self.sid, dec)

class TestCopy(Base):

    def test_copy(self):
        """ (H5S) Copy """
        sid = h5s.create_simple((10,10))
        sid.select_hyperslab((2,2), (5,5))
        sid2 = sid.copy()
        self.assertEqualSpaces(sid, sid2)
        self.assert_(sid is not sid2)















