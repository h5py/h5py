
"""
    Dataset testing operations.

    Tests all dataset operations, including creation, with the exception of:

    1. Slicing operations for read and write, handled by module test_slicing
    2. Type conversion for read and write (currently untested)
"""

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
        self.assertEqual(dset.size, 63)
        dset = self.f.create_dataset('bar', (6, 10))
        self.assertEqual(dset.shape, (6, 10))
        self.assertEqual(dset.size, (60))

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

    def test_dataset_intermediate_group(self):
        """ Create dataset with missing intermediate groups """
        ds = self.f.create_dataset("/foo/bar/baz", shape=(10, 10), dtype='<i4')
        self.assertIsInstance(ds, h5py.Dataset)
        self.assertTrue("/foo/bar/baz" in self.f)

    def test_reshape(self):
        """ Create from existing data, and make it fit a new shape """
        data = np.arange(30, dtype='f')
        dset = self.f.create_dataset('foo', shape=(10,3), data=data)
        self.assertEqual(dset.shape, (10,3))
        self.assertArrayEqual(dset[...],data.reshape((10,3)))

    def test_appropriate_low_level_id(self):
        " Binding Dataset to a non-DatasetID identifier fails with ValueError "
        with self.assertRaises(ValueError):
            Dataset(self.f['/'].id)

    def test_create_bytestring(self):
        """ Creating dataset with byte string yields vlen ASCII dataset """
        
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

class TestAutoCreate(BaseDataset):

    """
        Feauture: Datasets auto-created from data produce the correct types
    """

    def test_vlen_bytes(self):
        """ Assignment of a byte string produces a vlen ascii dataset """
        self.f['x'] = b"Hello there"
        ds = self.f['x']
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertTrue(tid.is_variable_str())
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)

    def test_vlen_unicode(self):
        """ Assignment of a unicode string produces a vlen unicode dataset """
        self.f['x'] = u"Hello there\u2034"
        ds = self.f['x']
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertTrue(tid.is_variable_str())
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)

    def test_string_fixed(self):
        """ Assignement of fixed-length byte string produces a fixed-length
        ascii dataset """
        self.f['x'] = np.string_("Hello there")
        ds = self.f['x']
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_size(), 11)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)

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

class TestStrings(BaseDataset):

    """
        Feature: Datasets created with vlen and fixed datatypes correctly
        translate to and from HDF5
    """

    def test_vlen_bytes(self):
        """ Vlen bytes dataset maps to vlen ascii in the file """
        dt = h5py.special_dtype(vlen=bytes)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)

    def test_vlen_unicode(self):
        """ Vlen unicode dataset maps to vlen utf-8 in the file """
        dt = h5py.special_dtype(vlen=unicode)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)

    def test_fixed_bytes(self):
        """ Fixed-length bytes dataset maps to fixed-length ascii in the file
        """
        dt = np.dtype("|S10")
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertFalse(tid.is_variable_str())
        self.assertEqual(tid.get_size(),10)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)

    def test_fixed_unicode(self):
        """ Fixed-length unicode datasets are unsupported (raise TypeError) """
        dt = np.dtype("|U10")
        with self.assertRaises(TypeError):
            ds = self.f.create_dataset('x', (100,), dtype=dt)

    def test_roundtrip_vlen_bytes(self):
        """ writing and reading to vlen bytes dataset preserves type and content
        """
        dt = h5py.special_dtype(vlen=bytes)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = b"Hello\xef"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), bytes)
        self.assertEqual(out, data)
        
    def test_roundtrip_vlen_unicode(self):
        """ Writing and reading to unicode dataset preserves type and content
        """
        dt = h5py.special_dtype(vlen=unicode)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = u"Hello\u2034"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), unicode)
        self.assertEqual(out, data)

    def test_roundtrip_fixed_bytes(self):
        """ Writing to and reading from fixed-length bytes dataset preserves
        type and content """
        dt = np.dtype("|S10")
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = b"Hello\xef"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), np.string_)
        self.assertEqual(out, data)

    @ut.expectedFailure
    def test_unicode_write_error(self):
        """ Writing a non-utf8 byte string to a unicode vlen dataset raises
        ValueError """
        dt = h5py.special_dtype(vlen=unicode)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = "Hello\xef"
        with self.assertRaises(ValueError):
            ds[0] = data

    def test_unicode_write_bytes(self):
        """ Writing valid utf-8 byte strings to a unicode vlen dataset is OK
        """
        dt = h5py.special_dtype(vlen=unicode)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = u"Hello there\u2034"
        ds[0] = data.encode('utf8')
        out = ds[0]
        self.assertEqual(type(out), unicode)
        self.assertEqual(out, data)

class TestCompound(BaseDataset):

    """
        Feature: Compound types correctly round-trip
    """

    def test_rt(self):
        """ Compound types are read back in correct order (issue 236)"""

        dt = np.dtype( [ ('weight', np.float64),
                             ('cputime', np.float64),
                             ('walltime', np.float64),
                             ('parents_offset', np.uint32),
                             ('n_parents', np.uint32),
                             ('status', np.uint8),
                             ('endpoint_type', np.uint8), ] )

        testdata = np.ndarray((16,),dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random((16,))*100

        self.f['test'] = testdata
        outdata = self.f['test'][...]
        self.assertTrue(np.all(outdata == testdata))
        self.assertEqual(outdata.dtype, testdata.dtype)

class TestEnum(BaseDataset):

    """
        Feature: Enum datatype info is preserved, read/write as integer
    """

    EDICT = {'RED': 0, 'GREEN': 1, 'BLUE': 42}

    def test_create(self):
        """ Enum datasets can be created and type correctly round-trips """
        dt = h5py.special_dtype(enum=('i', self.EDICT))
        ds = self.f.create_dataset('x', (100,100), dtype=dt)
        dt2 = ds.dtype
        dict2 = h5py.check_dtype(enum=dt2)
        self.assertEqual(dict2,self.EDICT)

    def test_readwrite(self):
        """ Enum datasets can be read/written as integers """
        dt = h5py.special_dtype(enum=('i4', self.EDICT))
        ds = self.f.create_dataset('x', (100,100), dtype=dt)
        ds[35,37] = 42
        ds[1,:] = 1
        self.assertEqual(ds[35,37], 42)
        self.assertArrayEqual(ds[1,:], np.array((1,)*100,dtype='i4'))










