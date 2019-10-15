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

