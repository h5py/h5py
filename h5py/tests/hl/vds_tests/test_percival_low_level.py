'''
Unit test for the low level vds interface for percival
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


import unittest
import numpy as np
import h5py as h5
import tempfile

class PercivalLowLevelTest(unittest.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5','raw_file_2.h5','raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = self.working_dir + outfile
            print "done file:"+filename
            f = h5.File(filename,'w')
            f['data'] = np.ones((20,200,200))*k
            k +=1
            f.close()
            
        f = h5.File(self.working_dir+'raw_file_4.h5','w')
        f['data'] = np.ones((19,200,200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        print "made the files"
        f.close()

    def test_percival_low_level(self):
        self.outfile = self.working_dir + 'percival.h5'
        print self.outfile
        with h5.File(self.outfile, 'w', libver='latest') as f:
            vdset_shape = (1,200,200)
            num = h5.h5s.UNLIMITED
            vdset_max_shape = (num,)+vdset_shape[1:]
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([-1]))
            # Create the source dataset dataspace
            k = 0
            for foo in self.fname:
                in_data = h5.File(foo)['data']
                src_shape = in_data.shape
                max_src_shape = (num,)+src_shape[1:]
                in_data.file.close()
                src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
                # Select the source dataset hyperslab
                src_dspace.select_hyperslab(start=(0, 0, 0),
                                            stride=(1,1,1),
                                            count=(num, 1, 1),
                                            block=(1,)+src_shape[1:])
                
                virt_dspace.select_hyperslab(start=(k, 0, 0),
                                             stride=(4,1,1),
                                             count=(num, 1, 1),
                                             block=(1,)+src_shape[1:])
        
                dcpl.set_virtual(virt_dspace, foo, 'data', src_dspace)
                k+=1
        
            # Create the virtual dataset
            dset = h5.h5d.create(f.id, name="data", tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)

            f = h5.File(self.outfile,'r')
            sh = f['data'].shape
            line = f['data'][:8,100,100]
            foo = np.array(2*range(4))
            f.close()
            self.assertEqual(sh,(79,200,200),)
            np.testing.assert_array_equal(line,foo)
            
    def tearDown(self):
        print "tearing down"
        import os
        for f in self.fname:
            print f
            os.remove(f)
        os.remove(self.outfile)
        print self.outfile


if __name__ == "__main__":
    unittest.main()
