import numpy as np
import h5py
from h5py import filters

import unittest

from common import res

class TestFilters(unittest.TestCase):

    def setUp(self):
        self.f = h5py.File(res.get_name(), 'w')

    def tearDown(self):
        res.clear()

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
            self.assertEqual(bool(dset.chunks), result)

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
                self.assertEqual(dset.chunks, chunk)

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
            msg =  'compression "%s"' % s
            dset = self.make_dset(compression=s)
            self.assertEqual(dset.compression, r, msg)
            self.assertEqual(dset.compression_opts, o, msg)

    def test_compression_opts(self):
        # Dataset compression keywords & options

        types = ('gzip', 'lzf')
        opts = {'gzip': (0, 9, 5), 'lzf': (None,)}

        if 'szip' in filters.encode:
            types += ('szip',)
            opts.update({'szip': (('nn', 4), ('ec', 8))})

        for t in types:
            for o in opts[t]:
                msg = "compression %s %s" % (t, o)
                dset = self.make_dset(compression=t, compression_opts=o)
                self.assertEqual(dset.compression, t, msg)
                self.assertEqual(dset.compression_opts, o, msg)

    def test_fletcher32_shuffle(self):
        # Check fletcher32 and shuffle

        settings = (None, False, True)
        results = (False, False, True)

        for s, r in zip(settings, results):
            msg = "test %s %s" % (s,r)
            dset = self.make_dset(fletcher32=s)
            self.assertEqual(dset.fletcher32, r, msg)
            dset = self.make_dset(shuffle=s)
            self.assertEqual(dset.shuffle, r, msg)

    def test_data(self):
        # Ensure data can be read/written with filters

        compression = (None, 'gzip', 'lzf')
        shapes = ((), (10,), (10,10), (200,200))
        # Filter settings should be ignored for scalar shapes

        types = ('f','i', 'c')

        def test_dset(shape, dtype, **kwds):
            msg = "test %s %s %s" % (shape, dtype, kwds)

            dset = self.make_dset(s, dtype, **kwds)
            arr = (np.random.random(s)*100).astype(dtype)
            dset[...] = arr
            assert np.all(dset[...] == arr), msg

        for s in shapes:
            for t in types:
                for c in compression:
                    test_dset(s, t, compression=c, shuffle=True)
                    test_dset(s, t, compression=c, shuffle=False)
                test_dset(s, t, fletcher32=True)
                test_dset(s, t, shuffle=True)
            









        
