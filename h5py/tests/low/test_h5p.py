
from h5py import tests
from h5py import *

config = h5.get_config()

class Base(tests.HTest):
    pass

if config.API_18:

    class TestLAID(Base):

        def setUp(self):
            self.id = h5p.create(h5p.LINK_ACCESS)

        def test_elink_fapl(self):
            """ (H5P) elink fapl """
            fapl = h5p.create(h5p.FILE_ACCESS)
            self.id.set_elink_fapl(fapl)
            self.assertEqual(self.id.get_elink_fapl(), fapl)

