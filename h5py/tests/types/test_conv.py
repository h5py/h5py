
import numpy as np
import tempfile
import h5py
from h5py import tests

class Base(tests.HTest):

    def setUp(self):
        self.f = h5py.File(tempfile.mktemp(), 'w', driver='core', backing_store=False)

    def tearDown(self):
        self.f.close()

