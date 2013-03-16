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

class TestPL(ut.TestCase):

    """
        Test track times
    """

    def test_obj_track_times(self):
        """ Tests if the object track times  set/get """
        # test for groups
        gcid = h5p.create(h5p.GROUP_CREATE)
        gcid.set_obj_track_times(False)
        self.assertEqual(False, gcid.get_obj_track_times())

        gcid.set_obj_track_times(True)
        self.assertEqual(True, gcid.get_obj_track_times())
        # test for datasets
        dcid = h5p.create(h5p.DATASET_CREATE)
        dcid.set_obj_track_times(False)
        self.assertEqual(False, dcid.get_obj_track_times())

        dcid.set_obj_track_times(True)
        self.assertEqual(True, dcid.get_obj_track_times())

        # test for generic objects
        ocid = h5p.create(h5p.OBJECT_CREATE)
        ocid.set_obj_track_times(False)
        self.assertEqual(False, ocid.get_obj_track_times())

        ocid.set_obj_track_times(True)
        self.assertEqual(True, ocid.get_obj_track_times())


