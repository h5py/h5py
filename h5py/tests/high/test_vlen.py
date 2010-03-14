
from h5py import tests
import h5py
import numpy as np
class TestCreate(tests.HTest):

    def test_dtype(self):
        """ (Vlen) Dtype round-trip """
        dt = h5py.special_dtype(vlen=str)
        self.assertEqual(h5py.check_dtype(vlen=dt), str)

class TestReadWrite(tests.HTest):

    def setUp(self):
        self.f = h5py.File('core', 'w', driver='core', backing_store=False)
    
    def tearDown(self):
        self.f.close()

    def test_read_attr(self):
        """ (Vlen) Attribute read """
        f = h5py.File(tests.getpath('vlstra.h5'),'r')
        try:
            self.assertEqual(f.attrs['test_scalar'], "This is the string for the attribute")
        finally:
            f.close()

    def test_write_attr(self):
        """ (Vlen) Attribute write """
        dt = h5py.special_dtype(vlen=str)
        self.f.attrs.create('vlen_attr', data="Hello there", dtype=dt)
        self.assertEqual(self.f.attrs['vlen_attr'], "Hello there")

    def test_read_dset(self):
        """ (Vlen) Dataset read """
        f = h5py.File(tests.getpath('strings.h5'), 'r')
        try:
            refarr = np.array(["A fight is a contract that takes two people to honor.",
                               "A combative stance means that you've accepted the contract.",
                               "In which case, you deserve what you get.",
                               "  --  Professor Cheng Man-ch'ing"], dtype="O")

            dset = f["StringsEx"]

            self.assert_(np.all(refarr == dset[...]), "Failed:\n%s" % "\n".join(str(x) for x in dset))
            self.assert_(np.all(refarr[2] == dset[2]), str(dset[2]))
            self.assert_(np.all(refarr[1:3] == dset[1:3]), str(dset[1:3]))

            self.assertEqual(dset.dtype, h5py.special_dtype(vlen=str))
        finally:
            f.close()

    def test_write_dset(self):
        """ (Vlen) Dataset write """

        dt = h5py.special_dtype(vlen=str)

        data_arr = np.array(["Hello there!", "string 2", "", "!!!!!"], dtype=dt)

        slices = [np.s_[0], np.s_[1:3], np.s_[...]]

        dset = self.f.create_dataset("vlen_ds", (4,), dt)
        for s in slices:
            dset[s] = data_arr[s]
            self.assert_(np.all(dset[s] == data_arr[s]), "slc %s data %s" % (s, data_arr[s]))

    @tests.require(h5py.version.hdf5_version_tuple >= (1,8,3))
    def test_compound(self):
        """ (Vlen) VL strings in compound types """

        vlen_dt = h5py.special_dtype(vlen=str)
        dts = [ [('a_name','>i4'), ('vlen',vlen_dt), ('d_name', '>f4')],
                [('a_name','=i8'), ('vlen',vlen_dt), ('d_name', '>f4')] ]

        for dt in dts:
            if 'vlen_ds' in self.f:
                del self.f['vlen_ds']
            data = np.ndarray((1,),dtype=dt)
            data['a_name'] = 42
            data['vlen'] = 'This is a variable-length string'
            data['d_name'] = 34.5
            dset = self.f.create_dataset("vlen_ds", data=data)
            self.assert_(np.all(dset[...] == data))

    def test_array(self):
        """ (Vlen) VL strings in array types """

        data = [["Hi", ""],["Hello there", "A string"]]

        vlen_dt = h5py.special_dtype(vlen=str)
        arr_dt = np.dtype((vlen_dt, (2,2)))

        arr = np.ndarray((20,), dtype=arr_dt)

        arr[0,:] = data

        ds = self.f.create_dataset('ds', data=arr)

        self.assert_(np.all(ds[0] == arr[0]))

