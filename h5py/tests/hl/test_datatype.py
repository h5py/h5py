"""
    Tests for the h5py.Datatype class.
"""

from __future__ import absolute_import

import numpy as np
import h5py

from ..common import ut, TestCase, NUMPY_RELEASE_VERSION

class TestVlen(TestCase):

    """
        Check that storage of vlen strings is carried out correctly.
    """
    
    def test_compound(self):

        fields = []
        fields.append(('field_1', h5py.special_dtype(vlen=str)))
        fields.append(('field_2', np.int32))
        dt = np.dtype(fields)
        self.f['mytype'] = np.dtype(dt)
        dt_out = self.f['mytype'].dtype.fields['field_1'][0]
        self.assertEqual(h5py.check_dtype(vlen=dt_out), str)
        
    def test_vlen_enum(self):
        fname = self.mktemp()
        arr1 = [[1],[1,2]]
        dt1 = h5py.special_dtype(vlen=h5py.special_dtype(
            enum=('i', dict(foo=1, bar=2))))

        with h5py.File(fname,'w') as f:
            df1 = f.create_dataset('test', (len(arr1),), dtype=dt1)
            df1[:] = np.array(arr1)

        with h5py.File(fname,'r') as f:
            df2  = f['test']
            dt2  = df2.dtype
            arr2 = [e.tolist() for e in df2[:]]

        self.assertEqual(arr1, arr2)
        self.assertEqual(h5py.check_dtype(enum=h5py.check_dtype(vlen=dt1)),
                         h5py.check_dtype(enum=h5py.check_dtype(vlen=dt2)))

@ut.skipIf(NUMPY_RELEASE_VERSION < (1, 11), "Requires numpy>=1.11")
class TestAligned(TestCase):

    """
        Check that offsets are correctly computed for aligned compound types.
    """

    def test_aligned_offsets(self):
        dt = np.dtype('i2,i8', align=True)
        ht = h5py.h5t.py_create(dt)
        self.assertEqual(dt.itemsize, ht.get_size())
        self.assertEqual(
            [dt.fields[i][1] for i in dt.names],
            [ht.get_member_offset(i) for i in range(ht.get_nmembers())]
        )

    def test_aligned_data(self):
        dt = np.dtype('i2,f8', align=True)
        data = np.empty(10, dtype=dt)

        data['f0'] = np.random.randint(-100, 100, size=data.size, dtype='i2')
        data['f1'] = np.random.rand(data.size)

        fname = self.mktemp()

        with h5py.File(fname, 'w') as f:
            f['data'] = data

        with h5py.File(fname, 'r') as f:
            self.assertArrayEqual(f['data'], data)
