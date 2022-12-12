# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
import h5py

class AttributeSuite:

    def setup(self):
        self._td = TemporaryDirectory()
        self.path = osp.join(self._td.name, '_attribute.h5')
        self.f = h5py.File(self.path, 'w')
        self.data = np.zeros(4)
        self.f.attrs.create('a', self.data)

    def teardown(self):
        self.f.close()
        self._td.cleanup()

    def time_create_attribute(self):
        for n in range(65535):
            self.f.attrs.create('b', self.data)

    def time_modify_attribute(self):
        self.f.attrs.modify('a', np.ones(4))
