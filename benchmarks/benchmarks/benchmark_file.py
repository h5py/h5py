# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class FileSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_file.h5')
        self.new_path = osp.join(self._td.name, '_newfile.h5')
        h5py.File(self.path, 'w')

    def teardown(self):
        self._td.cleanup()

    def time_create_file(self):
        with h5py.File(self.new_path, 'w') as f:
            pass

    def time_read_file(self):
        with h5py.File(self.path, 'r') as f:
            pass
