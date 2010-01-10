
"""
    Variable-length data types
"""

import numpy as np
import h5py

import unittest
from common import res

class TestVlen(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        res.clear()

    def test_create(self):
        dt = h5py.new_vlen(str)
        self.assertEqual(str, h5py.get_vlen(dt))
        self.assertEqual(dt.kind, "O")

    def test_read_attr(self):
        
        f = h5py.File(res.get_data_path('vlstra.h5'), 'r')

        self.assertEqual(f.attrs['test_scalar'], "This is the string for the attribute")

        aid = h5py.h5a.open(f.id, 'test_scalar')
        self.assertEqual(aid.dtype, h5py.new_vlen(str))

    def test_write_attr(self):

        f = h5py.File(res.get_name(),'w')

        value = "This is the string!"
        
        dt = h5py.new_vlen(str)
        f.attrs.create('test_string', value, dtype=dt)
        self.assertEqual(f.attrs['test_string'], value)
    
        aid = h5py.h5a.open(f.id, 'test_string')
        self.assertEqual(dt, aid.dtype)         


    def test_read_strings(self):

        f = h5py.File(res.get_data_path('strings.h5'), 'r')

        refarr = np.array(["A fight is a contract that takes two people to honor.",
                           "A combative stance means that you've accepted the contract.",
                           "In which case, you deserve what you get.",
                           "  --  Professor Cheng Man-ch'ing"], dtype="O")

        dset = f["StringsEx"]

        self.assert_(np.all(refarr == dset[...]), "Failed:\n%s" % "\n".join(str(x) for x in dset))
        self.assert_(np.all(refarr[2] == dset[2]), str(dset[2]))
        self.assert_(np.all(refarr[1:3] == dset[1:3]), str(dset[1:3]))

        self.assertEqual(dset.dtype, h5py.new_vlen(str))
    
    def test_write_strings(self):

        f = h5py.File(res.get_name(), 'w')

        dt = h5py.new_vlen(str)

        data_arr = np.array(["Hello there!", "string 2", "", "!!!!!"], dtype=dt)

        slices = [np.s_[0], np.s_[1:3], np.s_[...]]

        dset = f.create_dataset("vlen_ds", (4,), dt)
        for s in slices:
            dset[s] = data_arr[s]
            self.assert_(np.all(dset[s] == data_arr[s]), "slc %s data %s" % (s, data_arr[s]))


    def test_compound(self):

        vlen_dt = h5py.new_vlen(str)
        dts = [ [('a_name','>i4'), ('vlen',vlen_dt), ('d_name', '>f4')],
                [('a_name','=i8'), ('vlen',vlen_dt), ('d_name', '>f4')] ]

        f = h5py.File(res.get_name(), 'w')

        for dt in dts:
            if 'vlen_ds' in f:
                del f['vlen_ds']
            data = np.ndarray((1,),dtype=dt)
            data['a_name'] = 42
            data['vlen'] = 'This is a variable-length string'
            data['d_name'] = 34.5
            dset = f.create_dataset("vlen_ds", data=data)
            self.assert_(np.all(dset[...] == data))

    def test_array(self):

        f = h5py.File(res.get_name(), 'w')

        data = [["Hi", ""],["Hello there", "A string"]]

        vlen_dt = h5py.new_vlen(str)
        arr_dt = np.dtype((vlen_dt, (2,2)))

        arr = np.ndarray((20,), dtype=arr_dt)

        arr[0,:] = data

        ds = f.create_dataset('ds', data=arr)

        self.assert_(np.all(ds[0] == arr[0]))
















