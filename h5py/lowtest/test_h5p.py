try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from h5py import h5p, h5f

class TestLibver(ut.TestCase):

    """
        Feature: Setting/getting lib ver bounds
    """

    def test_libver(self):
        """ Test libver bounds set/get """
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_libver_bounds(h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST)
        self.assertEqual((h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST),
                         plist.get_libver_bounds())


class TestDA(ut.TestCase):
    '''
    Feature: setting/getting chunk cache size on a dataset access property list
    '''
    def test_chuck_cache(self):
        '''test get/set chunk cache '''
        dalist = h5p.create(h5p.DATASET_ACCESS)
        nslots = 10000 # 40kb hash table
        nbytes = 1000000 #1MB cache size
        w0 = .5 # even blend of eviction strategy

        dalist.set_chunk_cache(nslots, nbytes, w0)
        self.assertEqual((nslots, nbytes, w0),
                         dalist.get_chunk_cache())

class TestPL(ut.TestCase):
    def test_obj_track_times(self):
        """
        tests if the object track times  set/get
        """
        # test for groups
        gcid = h5p.create(h5p.GROUP_CREATE)
        gcid.set_obj_track_times(False)
        self.assertEqual(False,gcid.get_obj_track_times())

        gcid.set_obj_track_times(True)
        self.assertEqual(True,gcid.get_obj_track_times())
        # test for datasets
        dcid = h5p.create(h5p.DATASET_CREATE)
        dcid.set_obj_track_times(False)
        self.assertEqual(False,dcid.get_obj_track_times())

        dcid.set_obj_track_times(True)
        self.assertEqual(True,dcid.get_obj_track_times())

        # test for generic objects
        ocid = h5p.create(h5p.OBJECT_CREATE)
        ocid.set_obj_track_times(False)
        self.assertEqual(False,ocid.get_obj_track_times())

        ocid.set_obj_track_times(True)
        self.assertEqual(True,ocid.get_obj_track_times())

    def test_link_creation_tracking(self):
        """
        tests the link creation order set/get
        """

        gcid = h5p.create(h5p.GROUP_CREATE)
        gcid.set_link_creation_order(0)
        self.assertEqual(0, gcid.get_link_creation_order())

        flags = h5p.CRT_ORDER_TRACKED|h5p.CRT_ORDER_INDEXED
        gcid.set_link_creation_order(flags)
        self.assertEqual(flags, gcid.get_link_creation_order())

        # test for file creation
        fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_link_creation_order(flags)
        self.assertEqual(flags, fcpl.get_link_creation_order())

