"""
    Tests for the h5py.Datatype class.
"""

from __future__ import absolute_import

import numpy as np
import h5py

from ..common import ut, TestCase

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
        
