# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.AttributeManager.create() method.
"""

from __future__ import absolute_import

import numpy as np
import h5py

from ..common import ut, TestCase

class TestArray(TestCase):

    """
        Check that top-level array types can be created and read.
    """
    
    def test_int(self):
        # See issue 498
        
        dt = np.dtype('(3,)i')
        data = np.arange(3, dtype='i')
        
        self.f.attrs.create('x', data=data, dtype=dt)
        
        aid = h5py.h5a.open(self.f.id, b'x')
        
        htype = aid.get_type()
        self.assertEqual(htype.get_class(), h5py.h5t.ARRAY)
        
        out = self.f.attrs['x']
        
        self.assertArrayEqual(out, data)
        
    def test_string_dtype(self):
        # See issue 498 discussion
        
        self.f.attrs.create('x', data=42, dtype='i8')
