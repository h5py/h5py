

from unittest import TestCase
import numpy as np

from h5py import h5t
import ctypes

strings = ["Hi", "Hello", "This is a string", "HDF5 is awesome!"]
vlen_dtype = h5t.special_dtype(vlen=str)
vlen_htype = h5t.py_create(vlen_dtype, logical=1)
obj_htype = h5t.py_create(vlen_dtype)

class TestVlenObject(TestCase):

    """
        Test conversion routines between string vlens and object pointers
    """

    def test_obj2vlen_simple(self):
        """ Object to vlen (contiguous) """

        objarr = np.array(strings, dtype=vlen_dtype)

        destbuffer = np.ndarray(objarr.shape, dtype=np.uintp, buffer=objarr).copy()

        h5t.convert(obj_htype, vlen_htype, len(strings), destbuffer)

        for idx, val in enumerate(destbuffer):
            self.assertEqual(ctypes.string_at(int(val)), strings[idx])
        
    def test_obj2vlen_complex(self):
        """ Object to vlen (compound) """

        obj_ptr_size = h5t.PYTHON_OBJECT.get_size()
        vlen_ptr_size = vlen_htype.get_size()

        input_htype = h5t.create(h5t.COMPOUND, obj_ptr_size+4)
        input_htype.insert('a', 0, obj_htype)
        input_htype.insert('b', obj_ptr_size, h5t.STD_I32LE)

        output_htype = h5t.create(h5t.COMPOUND, vlen_ptr_size+4)
        output_htype.insert('a', 0, vlen_htype)
        output_htype.insert('b', vlen_ptr_size, h5t.STD_I32LE)

        objarr = np.ndarray((len(strings),), dtype=[('a', vlen_dtype), ('b', '<i4')])
        objarr['a'] = strings

        destbuffer = np.ndarray(objarr.shape, dtype=[('a', np.uintp), ('b', '<i4')], buffer=objarr).copy()

        h5t.convert(input_htype, output_htype, len(strings), destbuffer, destbuffer.copy())

        for idx, val in enumerate(destbuffer):
            self.assertEqual(ctypes.string_at(int(val[0])), strings[idx])







