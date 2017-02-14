'''
Unit test for the high level vds interface for eiger
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


import unittest
import numpy as np
import h5py as h5
import tempfile


class TestEigerHighLevel(unittest.TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5', 'raw_file_2.h5', 'raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = self.working_dir + outfile
            f = h5.File(filename, 'w')
            f['data'] = np.ones((20, 200, 200))*k
            k += 1
            f.close()

        f = h5.File(self.working_dir + 'raw_file_4.h5', 'w')
        f['data'] = np.ones((18, 200, 200)) * 3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        f.close()

    def test_eiger_high_level(self):
        self.outfile = self.working_dir + 'eiger.h5'
        TGT = h5.VirtualTarget(self.outfile, 'data', shape=(78, 200, 200))
        VMlist = []
        M_minus_1 = 0
        # Create the virtual dataset file
        with h5.File(self.outfile, 'w', libver='latest') as f:
            for foo in self.fname:
                in_data = h5.File(foo)['data']
                src_shape = in_data.shape
                in_data.file.close()
                M = M_minus_1 + src_shape[0]
                VSRC = h5.VirtualSource(foo, 'data', shape=src_shape)
                VM = h5.VirtualMap(VSRC, TGT[M_minus_1:M, :, :], dtype=float)
                VMlist.append(VM)
                M_minus_1 = M
            f.create_virtual_dataset(VMlist=VMlist, fillvalue=45)
            f.close()

        f = h5.File(self.outfile, 'r')['data']
        self.assertEqual(f[10, 100, 10], 0.0)
        self.assertEqual(f[30, 100, 100], 1.0)
        self.assertEqual(f[50, 100, 100], 2.0)
        self.assertEqual(f[70, 100, 100], 3.0)
        f.file.close()

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)


if __name__ == "__main__":
    unittest.main()
