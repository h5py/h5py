
"""
    Test NumPy dtype to HDF5 type conversion.
"""

import numpy as np
from h5py import tests, h5t, h5r

class TestIntegers(tests.HTest):

    def test_int(self):
        """ (Types) Integer dtype to HDF5 literal """
        htype = h5t.py_create('i')
        self.assertIsInstance(htype, h5t.TypeIntegerID)

    def test_int_log(self):
        """ (Types) Integer dtype to HDF5 logical """
        htype = h5t.py_create('i', logical=True)
        self.assertIsInstance(htype, h5t.TypeIntegerID)

    def test_enum_lit(self):
        """ (Types) Enum literal is HDF5 integer """
        dt = h5t.special_dtype(enum=('i', {'a': 1, 'b': 2}))
        htype = h5t.py_create(dt)
        self.assertIsInstance(htype, h5t.TypeIntegerID)

    def test_enum_log(self):
        """ (Types) Enum logical is HDF5 enum """
        dt = h5t.special_dtype(enum=('i', {'a': 1, 'b': 2}))
        htype = h5t.py_create(dt, logical=True)
        self.assertIsInstance(htype, h5t.TypeEnumID)

class TestFloats(tests.HTest):

    def test_float(self):
        """ (Types) Float dtype to HDF5 literal """
        htype = h5t.py_create('f')
        self.assertIsInstance(htype, h5t.TypeFloatID)

    def test_float_log(self):
        """ (Types) Float dtype to HDF5 logical """
        htype = h5t.py_create('f', logical=True)
        self.assertIsInstance(htype, h5t.TypeFloatID)

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

class TestArray(tests.HTest):

    def test_array(self):
        """ (Types) Multidimensional array type """
        htype = h5t.py_create(('f',(2,2)))
        self.assertIsInstance(htype, h5t.TypeArrayID)
        self.assertEqual(htype.get_array_dims(), (2,2))

    def test_array_dtype(self):
        """ (Types) Array dtypes using non-tuple shapes """
        dt1 = np.dtype('f4', (2,))
        dt2 = np.dtype('f4', [2])
        dt3 = np.dtype('f4', 2)
        dt4 = np.dtype('f4', 2.1)
        ht1 = h5t.py_create(dt1)
        ht2 = h5t.py_create(dt2)
        ht3 = h5t.py_create(dt3)
        ht4 = h5t.py_create(dt4)
        self.assertEqual(ht1.dtype, dt1)
        self.assertEqual(ht2.dtype, dt1)
        self.assertEqual(ht3.dtype, dt1)
        self.assertEqual(ht4.dtype, dt1)

class TestRef(tests.HTest):

    def test_objref(self):
        """ (Types) Object reference literal is Python object """
        dt = h5t.special_dtype(ref=h5r.Reference)
        htype = h5t.py_create(dt)
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

    def test_objref_log(self):
        """ (Types) Object reference logical is HDF5 reference """
        dt = h5t.special_dtype(ref=h5r.Reference)
        htype = h5t.py_create(dt, logical=True)
        self.assertEqual(htype, h5t.STD_REF_OBJ)

    def test_regref(self):
        """ (Types) Region reference literal is Python object """
        dt = h5t.special_dtype(ref=h5r.RegionReference)
        htype = h5t.py_create(dt)
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

    def test_regref_log(self):
        """ (Types) Region reference logical is HDF5 dset reference """
        dt = h5t.special_dtype(ref=h5r.RegionReference)
        htype = h5t.py_create(dt, logical=True)
        self.assertEqual(htype, h5t.STD_REF_DSETREG)

class TestComplex(tests.HTest):

    def test_complex(self):
        """ (Types) Simple complex creation """
        htype = h5t.py_create('c8')
        self.assertIsInstance(htype, h5t.TypeCompoundID)

class TestCompound(tests.HTest):

    def test_simple(self):
        """ (Types) Simple compound type (round-trip) """
        dt = np.dtype([('a','i'), ('b','f'),('c','f8')])
        htype = h5t.py_create(dt)
        self.assertEqual(htype.dtype, dt)

    def test_recursive(self):
        """ (Types) Compound type containing compound type (round-trip) """
        dt1 = np.dtype([('a','i'),('b','f')])
        dt2 = np.dtype([('a',dt1),('b','f8')])
        htype = h5t.py_create(dt2)
        self.assertEqual(htype.dtype, dt2)









