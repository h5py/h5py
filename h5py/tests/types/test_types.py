
"""
    Test NumPy dtype to HDF5 type conversion.
"""

import numpy as np
from h5py import tests, h5t

class TestIntegers(tests.HTest):

    def test_int(self):
        """ (Types) Integer dtype to HDF5 literal """
        htype = h5t.py_create('i')
        self.assertIsInstance(htype, h5t.TypeIntegerID)

    def test_int_log(self):
        """ (Types) Integer dtype to HDF5 logical """
        htype = h5t.py_create('i', logical=True)
        self.assertIsInstance(htype, h5t.TypeIntegerID)

class TestFloats(tests.HTest):

    def test_float(self):
        """ (Types) Float dtype to HDF5 literal """
        htype = h5t.py_create('f')
        self.assertIsInstance(htype, h5t.TypeFloatID)

    def test_float_log(self):
        """ (Types) Float dtype to HDF5 logical """
        htype = h5t.py_create('f', logical=True)

class TestString(tests.HTest):

    def test_string(self):
        """ (Types) String dtype to HDF5 literal """
        htype = h5t.py_create('S10')
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertEqual(htype.get_size(), 10)

    def test_string(self):
        """ (Types) String dtype to HDF5 logical """
        htype = h5t.py_create('S10', logical=True)
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertEqual(htype.get_size(), 10)

    def test_string(self):
        """ (Types) Length-1 string works OK """
        htype = h5t.py_create('S1')
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertEqual(htype.get_size(), 1)

    def test_vlstring_lit(self):
        """ (Types) Vlen string literal is Python object pointer """
        dt = h5t.special_dtype(vlen=str)
        htype = h5t.py_create(dt)
        self.assertIsInstance(htype, h5t.TypeOpaqueID)
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

    def test_vlstring_log(self):
        """ (Types) Vlen string logical is null-term HDF5 vlen ASCII string """
        dt = h5t.special_dtype(vlen=str)
        htype = h5t.py_create(dt, logical=True)
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertEqual(htype.is_variable_str(), True)
        self.assertEqual(htype.get_cset(), h5t.CSET_ASCII)
        self.assertEqual(htype.get_strpad(), h5t.STR_NULLTERM)








