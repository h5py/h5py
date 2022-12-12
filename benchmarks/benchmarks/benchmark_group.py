# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class TimeSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_group.h5')
        self.f = h5py.File(self.path, 'w')

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_create_group(self):
        self.f.create_group("group")
