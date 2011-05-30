import sys

import numpy as np

from .common import ut, TestCase
from h5py.highlevel import File, Group, Dataset
import h5py

class BaseDataset(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestRepr(BaseDataset):

    """
        Feature: repr(Dataset) behaves sensibly
    """
    
    def test_repr_open(self):
        """ repr() works on live and dead datasets """
        ds = self.f.create_dataset('foo', (4,))
        self.assertIsInstance(repr(ds), basestring)
        self.f.close()
        self.assertIsInstance(repr(ds), basestring)

class TestCreateShape(BaseDataset):

    """
        Feature: Datasets can be created from a shape only
    """

    def test_create_scalar(self):
        """ Create a scalar dataset """
        dset = self.f.create_dataset('foo', ())
        self.assertEqual(dset.shape, ())

    def test_create_simple(self):
        """ Create a size-1 dataset """
        dset = self.f.create_dataset('foo', (1,))
        self.assertEqual(dset.shape, (1,))

    def test_create_extended(self):
        """ Create an extended dataset """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.shape, (63,))

    def test_default_dtype(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.dtype, np.dtype('=f4'))

    def test_missing_shape(self):
        """ Missing shape raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo')

class TestCreateData(BaseDataset):

    """
        Feature: Datasets can be created from existing data
    """

    def test_create_scalar(self):
        """ Create a scalar dataset from existing array """
        data = np.ones((), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_create_extended(self):
        """ Create an extended dataset from existing data """
        data = np.ones((63,), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_reshape(self):
        """ Create from existing data, and make it fit a new shape """
        data = np.arange(30, dtype='f')
        dset = self.f.create_dataset('foo', shape=(10,3), data=data)
        self.assertEqual(dset.shape, (10,3))
        self.assertArrayEqual(dset[...],data.reshape((10,3)))

class TestCreateRequire(BaseDataset):

    """
        Feature: Datasets can be created only if they don't exist in the file
    """

    def test_create(self):
        """ Create new dataset with no conflicts """
        dset = self.f.require_dataset('foo', (10,3), 'f')
        self.assertIsInstance(dset, Dataset)
        self.assertEqual(dset.shape, (10,3))

    def test_create_existing(self):
        """ require_dataset yields existing dataset """
        dset = self.f.require_dataset('foo', (10,3), 'f')
        dset2 = self.f.require_dataset('foo',(10,3), 'f')
        self.assertEqual(dset, dset2)

    def test_shape_conflict(self):
        """ require_dataset with shape conflict yields TypeError """
        self.f.create_dataset('foo', (10,3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10,4), 'f')

    def test_type_confict(self):
        """ require_dataset with object type conflict yields TypeError """
        self.f.create_group('foo')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10,3), 'f')

    def test_dtype_conflict(self):
        """ require_dataset with dtype conflict (strict mode) yields TypeError
        """
        dset = self.f.create_dataset('foo', (10,3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10,3), 'S10')

    def test_dtype_close(self):
        """ require_dataset with convertible type succeeds (non-strict mode)
        """
        dset = self.f.create_dataset('foo', (10,3), 'i4')
        dset2 = self.f.require_dataset('foo', (10,3), 'i2', exact=False)
        self.assertEqual(dset, dset2)
        self.assertEqual(dset2.dtype, np.dtype('i4'))

class TestCreateChunked(BaseDataset):

    """
        Feature: Datasets can be created by manually specifying chunks
    """

    def test_create_chunks(self):
        """ Create via chunks tuple """
        dset = self.f.create_dataset('foo', shape=(100,), chunks=(10,))
        self.assertEqual(dset.chunks, (10,))

    def test_chunks_mismatch(self):
        """ Illegal chunk size raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', shape=(100,), chunks=(200,))

    def test_chunks_scalar(self):
        """ Attempting to create chunked scalar dataset raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo', shape=(), chunks=(50,))

    def test_auto_chunks(self):
        """ Auto-chunking of datasets """
        dset = self.f.create_dataset('foo', shape=(20,100), chunks=True)
        self.assertIsInstance(dset.chunks, tuple)
        self.assertEqual(len(dset.chunks), 2)

class TestCreateFillvalue(BaseDataset):

    """
        Feature: Datasets can be created with fill value
    """

    def test_create_fillval(self):
        """ Fill value is reflected in dataset contents """
        dset = self.f.create_dataset('foo', (10,), fillvalue=4.0)
        self.assertEqual(dset[0], 4.0)
        self.assertEqual(dset[7], 4.0)

    def test_property(self):
        """ Fill value is recoverable via property """
        dset = self.f.create_dataset('foo', (10,), fillvalue=3.0)
        self.assertEqual(dset.fillvalue, 3.0)
        self.assertNotIsInstance(dset.fillvalue, np.ndarray)

    def test_property_none(self):
        """ .fillvalue property works correctly if not set """
        dset = self.f.create_dataset('foo', (10,))
        self.assertEqual(dset.fillvalue, 0)

    def test_compound(self):
        """ Fill value works with compound types """
        dt = np.dtype([('a','f4'),('b','i8')])
        v = np.ones((1,), dtype=dt)[0]
        dset = self.f.create_dataset('foo', (10,), dtype=dt, fillvalue=v)
        self.assertEqual(dset.fillvalue, v)
        self.assertAlmostEqual(dset[4], v)

    def test_exc(self):
        """ Bogus fill value raises TypeError """
        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (10,),
                    dtype=[('a','i'),('b','f')], fillvalue=42)

@ut.skipIf('gzip' not in h5py.filters.encode, "DEFLATE is not installed")
class TestCreateGzip(BaseDataset):

    """
        Feature: Datasets created with gzip compression
    """

    def test_gzip(self):
        """ Create with explicit gzip options """
        dset = self.f.create_dataset('foo', (20,30), compression='gzip',
                                     compression_opts=9)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 9)

    def test_gzip_implicit(self):
        """ Create with implicit gzip level (level 4) """
        dset = self.f.create_dataset('foo', (20,30), compression='gzip')
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 4)

    def test_gzip_number(self):
        """ Create with gzip level by specifying integer """
        dset = self.f.create_dataset('foo', (20,30), compression=7)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 7)

    def test_gzip_exc(self):
        """ Illegal gzip level (explicit or implicit) raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20,30), compression=14)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20,30), compression=-4)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20,30), compression='gzip',
                                  compression_opts=14)


@ut.skipIf('lzf' not in h5py.filters.encode, "LZF is not installed")
class TestCreateLZF(BaseDataset):

    """
        Feature: Datasets created with LZF compression
    """

    def test_lzf(self):
        """ Create with explicit lzf """
        dset = self.f.create_dataset('foo', (20,30), compression='lzf')
        self.assertEqual(dset.compression, 'lzf')
        self.assertEqual(dset.compression_opts, None)

    def test_lzf_exc(self):
        """ Giving lzf options raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20,30), compression='lzf',
                                  compression_opts=4)


@ut.skipIf('szip' not in h5py.filters.encode, "SZIP is not installed")
class TestCreateSZIP(BaseDataset):

    """
        Feature: Datasets created with LZF compression
    """

    def test_szip(self):
        """ Create with explicit szip """
        dset = self.f.create_dataset('foo', (20,30), compression='szip',
                                     compression_opts=('ec',16))


@ut.skipIf('shuffle' not in h5py.filters.encode, "SHUFFLE is not installed")
class TestCreateShuffle(BaseDataset):

    """
        Feature: Datasets can use shuffling filter
    """

    def test_shuffle(self):
        """ Enable shuffle filter """
        dset = self.f.create_dataset('foo', (20,30), shuffle=True)
        self.assertTrue(dset.shuffle)


@ut.skipIf('fletcher32' not in h5py.filters.encode, "FLETCHER32 is not installed")
class TestCreateFletcher32(BaseDataset):    
    """
        Feature: Datases can use the fletcher32 filter
    """

    def test_fletcher32(self):
        """ Enable fletcher32 filter """
        dset = self.f.create_dataset('foo', (20,30), fletcher32=True)
        self.assertTrue(dset.fletcher32)


class TestResize(BaseDataset):

    """
        Feature: Datasets created with "maxshape" may be resized
    """

    def test_create(self):
        """ Create dataset with "maxshape" """
        dset = self.f.create_dataset('foo', (20,30), maxshape=(20,60))
        self.assertIsNot(dset.chunks, None)
        self.assertEqual(dset.maxshape, (20,60))

    def test_resize(self):
        """ Datasets may be resized up to maxshape """
        dset = self.f.create_dataset('foo', (20,30), maxshape=(20,60))
        self.assertEqual(dset.shape, (20,30))
        dset.resize((20,50))
        self.assertEqual(dset.shape, (20,50))
        dset.resize((20,60))
        self.assertEqual(dset.shape, (20,60))

    def test_resize_over(self):
        """ Resizing past maxshape triggers ValueError """
        dset = self.f.create_dataset('foo', (20,30), maxshape=(20,60))
        with self.assertRaises(ValueError):
            dset.resize((20,70))

    def test_resize_nonchunked(self):
        """ Resizing non-chunked dataset raises TypeError """
        dset = self.f.create_dataset("foo", (20,30))
        with self.assertRaises(TypeError):
            dset.resize((20,60))

    def test_resize_axis(self):
        """ Resize specified axis """
        dset = self.f.create_dataset('foo', (20,30), maxshape=(20,60))
        dset.resize(50, axis=1)
        self.assertEqual(dset.shape, (20,50))
     
    def test_axis_exc(self):
        """ Illegal axis raises ValueError """
        dset = self.f.create_dataset('foo', (20,30), maxshape=(20,60))
        with self.assertRaises(ValueError):
            dset.resize(50, axis=2)

    def test_zero_dim(self):
        """ Allow zero-length initial dims for unlimited axes (issue 111) """
        dset = self.f.create_dataset('foo', (15,0), maxshape=(15,None))
        self.assertEqual(dset.shape, (15,0))
        self.assertEqual(dset.maxshape, (15,None))

class TestDtype(BaseDataset):

    """
        Feature: Dataset dtype is available as .dtype property
    """

    def test_dtype(self):
        """ Retrieve dtype from dataset """
        dset = self.f.create_dataset('foo', (5,), '|S10')
        self.assertEqual(dset.dtype, np.dtype('|S10'))


class TestLen(BaseDataset):

    """
        Feature: Size of first axis is available via Python's len
    """

    def test_len(self):
        """ Python len() (under 32 bits) """
        dset = self.f.create_dataset('foo', (312,15))
        self.assertEqual(len(dset), 312)

    def test_len_big(self):
        """ Python len() vs Dataset.len() """
        dset = self.f.create_dataset('foo', (2**33,15))
        self.assertEqual(dset.shape, (2**33, 15))
        if sys.maxint == 2**31-1:
            with self.assertRaises(OverflowError):
                len(dset)
        else:
            self.assertEqual(len(dset), 2**33)
        self.assertEqual(dset.len(), 2**33)

class TestIter(BaseDataset):

    """
        Feature: Iterating over a dataset yields rows
    """

    def test_iter(self):
        """ Iterating over a dataset yields rows """
        data = np.arange(30, dtype='f').reshape((10,3))
        dset = self.f.create_dataset('foo', data=data)
        for x, y in zip(dset, data):
            self.assertEqual(len(x), 3)
            self.assertArrayEqual(x,y)

    def test_iter_scalar(self):
        """ Iterating over scalar dataset raises TypeError """
        dset = self.f.create_dataset('foo', shape=())
        with self.assertRaises(TypeError):
            [x for x in dset]





























