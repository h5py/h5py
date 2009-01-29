import numpy as np
import h5py
from h5py import filters

from nose.tools import assert_equal

from common import makehdf, delhdf

class TestFilters(object):

    def setUp(self):
        self.f = makehdf()

    def tearDown(self):
        delhdf(self.f)

    def make_dset(self, shape=None, dtype=None, **kwds):
        if 'dset' in self.f:
            del self.f['dset']
        if shape is None:
            shape = (10,10)
        if dtype is None:
            dtype = 'f'
        return self.f.create_dataset('dset', shape, dtype, **kwds)
        
    def test_chunks(self):
        # Check chunk behavior, including auto-chunking

        # Test auto-chunking
        pairs = [ ( {'chunks': None, 'compression': None}, False  ),
                  ( {'chunks': True, 'compression': None},  True  ),
                  ( {'chunks': None, 'compression': 'gzip'}, True ),
                  ( {'fletcher32': True}, True ),
                  ( {'shuffle': True}, True ),
                  ( {'maxshape': (None, None)}, True),
                  ( {}, False ) ]

        for kwds, result in pairs:
            dset = self.make_dset((10,10), **kwds)
            assert_equal(bool(dset.chunks), result)

        # Test user-defined chunking
        shapes = [(), (1,), (10,5), (1,10), (2**60, 2**60, 2**34)]
        chunks = {(): [None],
                  (1,): [None, (1,)],
                  (10,5): [None, (5,5), (10,1)],
                  (1,10): [None, (1,10), (1,3)],
                  (2**60, 2**60, 2**34): [(128, 64, 256)] }

        for shape in shapes:
            for chunk in chunks[shape]:
                dset = self.make_dset(shape, chunks=chunk)
                assert_equal(dset.chunks, chunk)

    def test_compression(self):
        # Dataset compression keywords only

        settings = (0, 9, 4, 'gzip', 'lzf', None)
        results  = ('gzip', 'gzip', 'gzip', 'gzip', 'lzf', None)
        opts     = (0, 9, 4, filters.DEFAULT_GZIP, None, None)
    
        if 'szip' in filters.encode:
            settings += ('szip',)
            results  += ('szip',)
            opts     += (filters.DEFAULT_SZIP,)

        for s, r, o in zip(settings, results, opts):
            print 'compression "%s"' % s
            dset = self.make_dset(compression=s)
            assert_equal(dset.compression, r)
            assert_equal(dset.compression_opts, o)

    def test_compression_opts(self):
        # Dataset compression keywords & options

        types = ('gzip', 'lzf')
        opts = {'gzip': (0, 9, 5), 'lzf': (None,)}

        if 'szip' in filters.encode:
            types += ('szip',)
            opts.update({'szip': (('nn', 4), ('ec', 8))})

        for t in types:
            for o in opts[t]:
                print "compression %s %s" % (t, o)
                dset = self.make_dset(compression=t, compression_opts=o)
                assert_equal(dset.compression, t)
                assert_equal(dset.compression_opts, o)

    def test_fletcher32_shuffle(self):
        # Check fletcher32 and shuffle, including auto-shuffle

        settings = (None, False, True)
        results = (False, False, True)

        for s, r in zip(settings, results):
            print "test %s %s" % (s,r)
            dset = self.make_dset(fletcher32=s)
            assert_equal(dset.fletcher32, r)
            dset = self.make_dset(shuffle=s)
            assert_equal(dset.shuffle, r)

        # Make sure shuffle is automatically activated for compression

        dset = self.make_dset(compression='gzip')
        assert_equal(dset.shuffle, True)
        dset = self.make_dset(compression='gzip', shuffle=False)
        assert_equal(dset.shuffle, False)

    def test_data(self):
        # Ensure data can be read/written with filters

        compression = (None, 'gzip', 'lzf')
        shapes = ((), (10,), (10,10), (200,200))
        # Filter settings should be ignored for scalar shapes

        types = ('f','i', 'c')

        def test_dset(shape, dtype, **kwds):
            print "test %s %s %s" % (shape, dtype, kwds)
            dset = self.make_dset(s, dtype, **kwds)
            arr = (np.random.random(s)*100).astype(dtype)
            dset[...] = arr
            assert np.all(dset[...] == arr)

        for s in shapes:
            for t in types:
                for c in compression:
                    test_dset(s, t, compression=c, shuffle=True)
                    test_dset(s, t, compression=c, shuffle=False)
                test_dset(s, t, fletcher32=True)
                test_dset(s, t, shuffle=True)
            









        
