# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class DataTypeSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_datatype.h5')
        self.f = h5py.File(self.path, 'w')
        self.dset = self.f.create_dataset('data', shape=(10,), dtype='f8')

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_type_conversion(self):
        data = self.dset.astype('i8')
