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
