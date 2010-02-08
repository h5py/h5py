from h5py import tests
from h5py import *

class TestCreate(tests.HTest):

    def setUp(self):
        self.fid, self.name = tests.gettemp()

    def tearDown(self):
        import os
        self.fid.close()
        os.unlink(self.name)

    @tests.require(api=18)
    def test_create_anon(self):
        """ (H5D) Anonymous dataset creation """
        sid = h5s.create_simple((10,10))
        dsid = h5d.create(self.fid, None, h5t.STD_I32LE, sid)
        self.assert_(dsid)
        self.assertIsInstance(dsid, h5d.DatasetID)
