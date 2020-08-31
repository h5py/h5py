# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class TimeSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        path = osp.join(self._td.name, 'test.h5')
        with h5py.File(path, 'w') as f:
            f['a'] = np.arange(100000)

        self.f = h5py.File(path, 'r')

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_many_small_reads(self):
        ds = self.f['a']
        for i in range(10000):
            arr = ds[i * 10:(i + 1) * 10]

class TimeWritingSuite:
    """Based on example in GitHub issue 492:
    https://github.com/h5py/h5py/issues/492
    """
    def setup(self):
        self._td = TemporaryDirectory()
        path = osp.join(self._td.name, 'test.h5')
        self.f = h5py.File(path, 'w')
        self.shape = shape = (128, 1024, 512)
        self.f.create_dataset(
            'a', shape=shape, dtype=np.float32, chunks=(1, shape[1], 64)
        )
        self.ds = self.f['a']
        self.data = np.zeros(self.shape[:2])
        self.datax = np.zeros(self.shape[1:])

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_write_index_first_axis(self):
        for i in range(self.shape[0]):
            self.ds[i,:,:] = self.datax

    def time_write_slice_first_axis(self):
        for i in range(self.shape[0]):
            self.ds[i:i+1,:,:] = self.datax[np.newaxis, ...]

    def time_write_index_last_axis(self):
        for i in range(self.shape[2]):
            self.ds[..., i] = self.data

    def time_write_slice_last_axis(self):
        for i in range(self.shape[2]):
            self.ds[..., i:i+1] = self.data[..., np.newaxis]


class TimeDatasetSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_dataset.h5')
        self.f = h5py.File(self.path, 'w')

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_create_dataset(self):
        self.f.create_dataset('dataset', (100,), dtype='f')

    def time_create_empty_dataset(self):
        self.f.create_dataset('empty', dtyp='f')

class TimeSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_operate.h5')
        self.f = h5py.File(self.path, 'w')

        self.dset = self.f.create_dataset('dset', (100,))
        self.arr = np.ones((100,))
        self.arr2 = np.ones((200,))

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_read_direct(self):
        self.dset.read_direct(self.arr, np.s_[0:10], np.s_[50:60])

    def time_write_direct(self):
        self.dset.write_direct(self.arr2, np.s_[0:10], np.s_[50:60])


class TimeVirtualDataset:

    def setup(self):
        self.layout = h5py.VirtualLayout(shape=(4,100), dtype='i4')
        for n in range(4):
            filename = "{}.h5".format(n)
            vsource = h5py.VirtualSource(filename, 'data', shape=(100,))
            self.layout[n] = vsource

        self.f = h5py.File('VDS.h5', 'w', libver='latest')

    def teardown(self):
        self.f.close()

    def time_virtual_dataset(self):
        self.f.create_virtual_dataset('vdata', self.layout, fillvalue=-1)

class TimeChunkedDataset:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, 'chunked_dataset.h5')
        self.f = h5py.File(self.path, 'w')

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_chunked_dataset(self):
        self.f.create_dataset('chunked', shape=(100,100,100), chunks=(10,10,10))

class TimeBroadcast:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, 'broadcast.h5')
        self.f = h5py.File(self.path, 'w')
        self.dset = self.f.create_dataset('3D',shape=(1024,512,1024))
        self.ds = self.f.create_dataset('2D',shape=(512,1024))

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_write_with_broadcast(self):
        self.dset[0] = self.ds

class TimeIndices:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_dataset.h5')
        self.f = h5py.File(self.path, 'w')
        self.dset = self.f.create_dataset('data', data= np.arange(512).reshape(8,8,8))
        self.indices = slice(2,8).indices(4)

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_write_with_indices(self):
        self.dset[self.indices] = 0
