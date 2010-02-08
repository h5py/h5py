
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
        """ (H5G) Anonymous group creation """
        gid = h5g.create(self.fid, None)
        self.assert_(gid)
        self.assertIsInstance(gid, h5g.GroupID)

