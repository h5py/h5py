
import numpy as np
import h5py
import unittest
import os.path as op
import os
from common import skip
from tempfile import mktemp

class TestVlen(unittest.TestCase):

    def test_create(self):
        dt = h5py.new_vlen(str)
        self.assertEqual(str, h5py.get_vlen(dt))
        self.assertEqual(dt.kind, "O")

    def test_read_attr(self):
        
        f = h5py.File(op.join(op.dirname(h5py.__file__), 'tests/data/vlstra.h5'), 'r')

        self.assertEqual(f.attrs['test_scalar'], "This is the string for the attribute")

        aid = h5py.h5a.open(f.id, 'test_scalar')
        self.assertEqual(aid.dtype, h5py.new_vlen(str))

    def test_write_attr(self):

        fname = mktemp('.hdf5')

        f = h5py.File(fname,'w')
        try:
            value = "This is the string!"
            
            dt = h5py.new_vlen(str)
            f.attrs.create('test_string', value, dtype=dt)
            self.assertEqual(f.attrs['test_string'], value)
        
            aid = h5py.h5a.open(f.id, 'test_string')
            self.assertEqual(dt, aid.dtype)
        finally:
            if f:
                f.close()
                os.unlink(fname)            


    def test_read_strings(self):

        f = h5py.File(op.join(op.dirname(h5py.__file__), 'tests/data/strings.h5'), 'r')

        refarr = np.array(["A fight is a contract that takes two people to honor.",
                           "A combative stance means that you've accepted the contract.",
                           "In which case, you deserve what you get.",
                           "  --  Professor Cheng Man-ch'ing"], dtype="O")

        print "\nReading vlen strings:\n"+"-"*60
        dset = f["StringsEx"]

        for idx, x in enumerate(dset):
            print '%d  "%s"' % (idx, x)
        print "-"*60

        self.assert_(np.all(refarr == dset[...]))
        self.assert_(np.all(refarr[2] == dset[2]))
        self.assert_(np.all(refarr[1:3] == dset[1:3]))

        self.assertEqual(dset.dtype, h5py.new_vlen(str))
    
    def test_write_strings(self):

        fname = mktemp('.hdf5')

        f = h5py.File(fname, 'w')
        try:
            dt = h5py.new_vlen(str)

            data_arr = np.array(["Hello there!", "string 2", "", "!!!!!"], dtype=dt)

            slices = [np.s_[0], np.s_[1:3], np.s_[...]]

            dset = f.create_dataset("vlen_ds", (4,), dt)
            for s in slices:
                print "slc %s data %s" % (s, data_arr[s])
                dset[s] = data_arr[s]
                self.assert_(np.all(dset[s] == data_arr[s]))
        finally:
            if f:
                f.close()
                os.unlink(fname)            


    def test_compound(self):

        vlen_dt = h5py.new_vlen(str)
        dts = [ [('a_name','>i4'), ('vlen',vlen_dt), ('d_name', '>f4')],
                [('a_name','=i8'), ('vlen',vlen_dt), ('d_name', '>f4')] ]

        fname = mktemp('.hdf5')
        f = h5py.File(fname, 'w')
        try:
            for dt in dts:
                if 'vlen_ds' in f:                 del f['vlen_ds']
                data = np.ndarray((1,),dtype=dt)
                data['a_name'] = 42
                data['vlen'] = 'This is a variable-length string'
                data['d_name'] = 34.5
                dset = f.create_dataset("vlen_ds", data=data)
                self.assert_(np.all(dset[...] == data))

        finally:
            if f:
                f.close()
                os.unlink(fname)

    def test_array(self):

        fname = mktemp('.hdf5')
        f = h5py.File(fname, 'w')

        data = [["Hi", ""],["Hello there", "A string"]]

        try:
            vlen_dt = h5py.new_vlen(str)
            arr_dt = np.dtype((vlen_dt, (2,2)))

            arr = np.ndarray((20,), dtype=arr_dt)

            arr[0,:] = data

            ds = f.create_dataset('ds', data=arr)

            self.assert_(np.all(ds[0] == arr[0]))
        finally:
            if f:
                f.close()
                os.unlink(fname)            


















