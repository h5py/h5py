import numpy as np
import h5py
from h5py import File
from .common import TestCase


class TestSlicedArrays(TestCase):
    """Sliced Arrays are stored correctly"""

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_sliced_array(self):
        self.f.create_dataset('x', (10,), dtype=h5py.vlen_dtype('bool'))
        x = np.array([True, False, True, True, False, False, False])
        self.f['x'][0] = x[::2]

        assert all(self.f['x'][0] == x[::2]), f"{self.f['x'][0]} != {x[::2]}"

        self.f.create_dataset('y', (10,), dtype=h5py.vlen_dtype('int8'))
        y = np.array([2, 4, 1, 5, -1, 3, 7])
        self.f['y'][0] = y[::2]

        assert all(self.f['y'][0] == y[::2]), f"{self.f['y'][0]} != {y[::2]}"
