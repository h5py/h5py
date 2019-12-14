# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
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

class WritingTimeSuite:
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

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_write_index_last_axis(self):
        ds = self.f['a']
        data = np.zeros(self.shape[:2])
        for i in range(self.shape[2]):
            ds[..., i] = data

    def time_write_slice_last_axis(self):
        ds = self.f['a']
        data = np.zeros(self.shape[:2])
        for i in range(self.shape[2]):
            ds[..., i:i+1] = data[..., np.newaxis]
