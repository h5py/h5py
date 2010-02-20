
from h5py import tests
from h5py import *

class Base(tests.HTest):

    def setUp(self):
        self.fid, self.name = tests.gettemp()
        
    def tearDown(self):
        import os
        self.fid.close()
        os.unlink(self.name)

class TestType(Base):

    def test_type(self):
        """ (H5I) get_type() returns typecode """
        x = h5i.get_type(self.fid)
        self.assertEqual(x, h5i.FILE)

    def test_exc(self):
        """ (H5I) get_type() returns BADID for closed object """
        g = h5g.open(self.fid, '/')
        g._close()
        self.assertEqual(h5i.get_type(g), h5i.BADID)

class TestName(Base):

    def test_name(self):
        """ (H5I) get_name() returns string name """
        g = h5g.create(self.fid, '/foobar')
        self.assertEqual(h5i.get_name(g), '/foobar')

    def test_noname(self):
        """ (H5I) get_name() returns None for unnamed & invalid objects """
        sid = h5s.create_simple((10,10))
        g = h5g.open(self.fid, '/')
        g._close()
        self.assertIsNone(h5i.get_name(sid))
        self.assertIsNone(h5i.get_name(g))

class TestFID(Base):

    def test_fid(self):
        """ (H5I) get_file_id() returns FID, with increased refcount """
        g = h5g.create(self.fid, '/foobar')
        x = h5i.get_file_id(g)
        self.assertIsInstance(x, h5f.FileID)
        self.assertEqual(x, self.fid)
        self.assertEqual(h5i.get_ref(x), 2)

    def test_exc(self):
        """ (H5I) get_file_id() on closed object raises ValueError """
        g = h5g.open(self.fid, '/')
        g._close()
        self.assertRaises(ValueError, h5i.get_file_id, g)

class TestRefs(tests.HTest):

    def test_refs(self):
        """ (H5I) get_ref, inc_ref, dec_ref are consistent """
        sid = h5s.create_simple((10,10))
        self.assertEqual(h5i.get_ref(sid), 1)
        
        h5i.inc_ref(sid)
        self.assertEqual(h5i.get_ref(sid), 2)

        h5i.dec_ref(sid)
        self.assertEqual(h5i.get_ref(sid), 1)








