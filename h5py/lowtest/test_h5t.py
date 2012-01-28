try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import sys
import numpy as np
import h5py
from h5py import h5t

class TestCompound(ut.TestCase):

    """
        Feature: Compound types can be created from Python dtypes
    """

    def test_ref(self):
        """ Reference types are correctly stored in compound types (issue 144)
        """
        ref = h5py.special_dtype(ref=h5py.Reference)
        dt = np.dtype([('a',ref),('b','<f4')])
        tid = h5t.py_create(dt,logical=True)
        t1, t2 = tid.get_member_type(0), tid.get_member_type(1)
        self.assertEqual(t1, h5t.STD_REF_OBJ)
        self.assertEqual(t2, h5t.IEEE_F32LE)
        self.assertEqual(tid.get_member_offset(0), 0)
        self.assertEqual(tid.get_member_offset(1), h5t.STD_REF_OBJ.get_size())

class TestStrings_Dtype2HDF5(ut.TestCase):

    """
        Tests dtype conversion rules governing Unicode and byte strings (Py2 and Py3)
    """

    @ut.expectedFailure
    def test_fixedbytes(self):
        """ Fixed-width byte string dtype to hdf5 type """
        dt = np.dtype("|S10")
 
        # For fixed-width strings the exact and logical representations are
        # identical.
        for htype in (h5t.py_create(dt), h5t.py_create(dt, logical=1)):
            self.assertIsInstance(htype, h5t.TypeStringID)
            self.assertFalse(htype.is_variable_str())
            self.assertEqual(htype.get_size(), 10)

    @ut.expectedFailure
    def test_vlenbytes(self):
        """ Vlen byte string dtype to hdf5 type """
        dt = h5py.special_dtype(vlen=bytes)

        # The exact representation is a size_t OPAQUE type
        htype = h5t.py_create(dt)
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

        # The logical representation is a variable-length string with CSET 0
        htype = h5t.py_create(dt, logical=1)
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertTrue(htype.is_variable_string())
        self.assertEqual(htype.get_cset(), h5t.CSET_ASCII)

    @ut.expectedFailure
    def test_fixedunicode(self):
        """ NumPy unicode string dtype to hdf5 type """
        dt = np.dtype("=U10")

        # The exact representation is a 4-byte OPAQUE type    
        htype = h5t.py_create(dt)
        self.assertEqual(htype, h5t.NUMPY_UNICODE)

        # The logical representation is a variable-length string with CSET 1
        htype = h5t.py_create(dt, logical=1)
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertTrue(htype.is_variable_str())
        self.assertEqual(htype.get_cset(), h5t.CSET_UTF8)

    @ut.expectedFailure
    def test_vlenunicode(self):
        """ Vlen unicode string to hdf5 type """
        dt = h5py.special_dtype(vlen=unicode)

        # The exact representation is a size_t OPAQUE type
        htype = h5t.py_create(dt)
        self.assertEqual(htype, h5t.PYTHON_OBJECT)

        # The logical representation is a variable-length string with CSET 1
        htype = h5t.py_create(dt, logical=1)
        self.assertIsInstance(htype, h5t.TypeStringID)
        self.assertTrue(htype.is_variable_string())
        self.assertEqual(htype.get_cset(), h5t.CSET_UTF8)

@ut.skipIf(sys.version_info[0] != 2, "Py2 only")
class TestStrings_HDF52Dtype_Py2(ut.TestCase):

    """
        Test HDF5 type to NumPy type (Py2)
    """

    @ut.expectedFailure
    def test_fixed_ascii(self):
        """ Fixed-ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_ASCII)

        dt = htype.py_dtype()
        self.assertEqual(dt, np.dtype("|S10"))

    @ut.expectedFailure
    def test_fixed_utf8(self):
        """ Fixed-utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_UTF8)

        dt = htype.py_dtype()
        self.assertEqual(dt, np.dtype("=U10"))

    @ut.expectedFailure
    def test_vlen_ascii(self):
        """ Vlen ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_ASCII)

        dt = htype.py_dtype()
        self.assertEqual(h5py.check_dtype(vlen=dt), bytes)

    @ut.expectedFailure
    def test_vlen_utf8(self):
        """ Vlen utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_UTF8)

        dt = htype.py_dtype()
        self.assertEqual(h5py.check_dtype(vlen=dt), unicode)

@ut.skipIf(sys.version_info[0] != 3, "Py3 only")
class TestStrings_HDF52Dtype_Py3(ut.TestCase):

    """
        Test HDF5 type to NumPy type (Py2)
    """

    @ut.expectedFailure
    def test_fixed_ascii(self):
        """ Fixed-ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_ASCII)

        dt = htype.py_dtype()
        self.assertEqual(dt, np.dtype("=U10"))

    @ut.expectedFailure
    def test_fixed_utf8(self):
        """ Fixed-utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_UTF8)

        dt = htype.py_dtype()
        self.assertEqual(dt, np.dtype("=U10"))

    @ut.expectedFailure
    def test_vlen_ascii(self):
        """ Vlen ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_ASCII)

        dt = htype.py_dtype()
        self.assertEqual(h5py.check_dtype(vlen=dt), unicode)

    @ut.expectedFailure
    def test_vlen_utf8(self):
        """ Vlen utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_UTF8)

        dt = htype.py_dtype()
        self.assertEqual(h5py.check_dtype(vlen=dt), unicode)

class TestStrings_ForceBytes(ut.TestCase):

    """
        Test HDF5 type to NumPy dtype with byte string forcing
    """

    @ut.expectedFailure
    def test_fixed_ascii(self):
        """ Fixed-ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_ASCII)

        with h5py.get_config().read_byte_strings:
            dt = htype.py_dtype()
            self.assertEqual(dt, np.dtype("=S10"))

    @ut.expectedFailure
    def test_fixed_utf8(self):
        """ Fixed-utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(10)
        htype.set_cset(h5t.CSET_UTF8)

        with h5py.get_config().read_byte_strings:
            dt = htype.py_dtype()
            self.assertEqual(dt, np.dtype("=S10"))

    @ut.expectedFailure
    def test_vlen_ascii(self):
        """ Vlen ascii to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_ASCII)

        with h5py.get_config().read_byte_strings:
            dt = htype.py_dtype()
            self.assertEqual(h5py.check_dtype(vlen=dt), bytes)

    @ut.expectedFailure
    def test_vlen_utf8(self):
        """ Vlen utf8 to NumPy dtype """
        htype = h5t.C_S1.copy()
        htype.set_size(h5t.VARIABLE)
        htype.set_cset(h5t.CSET_UTF8)

        with h5py.get_config().read_byte_strings:
            dt = htype.py_dtype()
            self.assertEqual(h5py.check_dtype(vlen=dt), bytes)





