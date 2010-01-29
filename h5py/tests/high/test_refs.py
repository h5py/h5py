

from h5py import tests
import h5py
import numpy as np

class Base(tests.HTest):

    def setUp(self):
        import tempfile
        self.name = tempfile.mktemp()
        self.f = h5py.File(self.name, 'w')

    def tearDown(self):
        import os
        self.f.close()
        os.unlink(self.name)


class TestObjRef(Base):

    def test_create(self):
        """ (Refs) Group reference round-trip """
        g = self.f.create_group('foo')
        ref = g.ref
        self.assertIsInstance(ref, h5py.Reference)
        x = self.f[ref]
        self.assertIsInstance(x, h5py.Group)
        self.assertEqual(x, g)

    def test_dtype(self):
        """ (Refs) Named type reference round-trip """
        dt = np.dtype('f')
        self.f['nt'] = dt
        nt = self.f['nt']
        ref = nt.ref
        self.assertIsInstance(ref, h5py.Reference)
        x = self.f[ref]
        self.assertIsInstance(x, h5py.Datatype)
        self.assertEqual(x, nt)

    @tests.require(api=18)
    def test_name(self):
        """ (Refs) .name property works on dereferenced objects (1.8) """
        g = self.f.create_group('foo')
        x = self.f[g.ref]
        self.assertEqual(x.name, g.name)

    @tests.require(api=16)
    def test_name_16(self):
        """ (Refs) .name property gives None for dereferenced objects (1.6) """
        g = self.f.create_group('foo')
        x = self.f[g.ref]
        self.assertIsNone(x.name)

    @tests.fixme("TypeError in h5r")
    def test_bool(self):
        """ (Refs) __nonzero__ tracks validity """
        ref = h5py.Reference()
        self.assert_(not ref)
        self.assert_(self.f.ref)

    @tests.fixme("TypeError in h5r")
    def test_exc(self):
        """ (Refs) Deref of empty ref raises ValueError """
        ref = h5py.Reference()
        self.assertRaises(ValueError, self.f.__getitem__, ref)

class TestRegRef(Base):

    def setUp(self):
        Base.setUp(self)
        self.ds = self.f.create_dataset('ds', (100,100), 'f')
        self.arr = np.arange(100*100, dtype='f').reshape((100,100))
        self.ds[...] = self.arr

    def test_create(self):
        """ (Refs) Region reference round-trip with Group """
        ref = self.ds.regionref[...]
        self.assertIsInstance(ref, h5py.RegionReference)
        x = self.f[ref]
        self.assertIsInstance(x, h5py.Dataset)
        self.assertEqual(x, self.ds)

    def test_sel(self):
        """ (Refs) Region selection can be used as slice (2D -> 1D) """
        ref = self.ds.regionref[10:20, 17:35]
        x = self.ds[ref]
        self.assertArrayEqual(x, self.arr[10:20, 17:35].flatten())





