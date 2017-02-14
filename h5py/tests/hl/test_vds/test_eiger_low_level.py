'''
Unit test for the low level vds interface for eiger
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


import unittest
import numpy as np
import h5py as h5
import tempfile


class EigerLowLevelTest(unittest.TestCase):
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

        f = h5.File(self.working_dir+'raw_file_4.h5', 'w')
        f['data'] = np.ones((18, 200, 200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        f.close()

    def test_eiger_low_level(self):
        self.outfile = self.working_dir + 'eiger.h5'
        with h5.File(self.outfile, 'w', libver='latest') as f:
            vdset_shape = (78, 200, 200)
            vdset_max_shape = vdset_shape
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([-1]))
            # Create the source dataset dataspace
            k = 0
            for foo in self.fname:
                in_data = h5.File(foo)['data']
                src_shape = in_data.shape
                max_src_shape = src_shape
                in_data.file.close()
                src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
                # Select the source dataset hyperslab
                src_dspace.select_hyperslab(start=(0, 0, 0),
                                            stride=(1, 1, 1),
                                            count=(1, 1, 1),
                                            block=src_shape)

                virt_dspace.select_hyperslab(start=(k, 0, 0),
                                             stride=(1, 1, 1),
                                             count=(1, 1, 1),
                                             block=src_shape)

                dcpl.set_virtual(virt_dspace, foo, 'data', src_dspace)
                k += src_shape[0]

            # Create the virtual dataset
            h5.h5d.create(f.id, name="data", tid=h5.h5t.NATIVE_INT16,
                          space=virt_dspace, dcpl=dcpl)

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
