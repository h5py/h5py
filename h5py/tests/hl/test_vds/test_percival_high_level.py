'''
Unit test for the high level vds interface for percival
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


import unittest
import numpy as np
import h5py as h5
import tempfile

class PercivalHighLevelTest(unittest.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5','raw_file_2.h5','raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = self.working_dir + outfile
            f = h5.File(filename,'w')
            f['data'] = np.ones((20,200,200))*k
            k +=1
            f.close()

        f = h5.File(self.working_dir+'raw_file_4.h5','w')
        f['data'] = np.ones((19,200,200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        f.close()

    def test_percival_high_level(self):
        self.outfile = self.working_dir+'percival.h5'
        VM=[]
        # Create the virtual dataset file
        with h5.File(self.outfile, 'w', libver='latest') as f:
            TGT = h5.VirtualTarget(self.outfile, 'data', shape=(79,200,200), maxshape=(None, 200,200)) # Virtual target is a representation of the output dataset
            k = 0
            for foo in self.fname:
                VSRC = h5.VirtualSource(foo, 'data',shape=(20,200,200),maxshape=(None, 200,200))
                VM.append(h5.VirtualMap(VSRC,TGT[k:79:4,:,:] , dtype=np.float))
                k+=1
            f.create_virtual_dataset(VMlist=VM, fillvalue=-5) # pass the fill value and list of maps
            f.close()

        f = h5.File(self.outfile,'r')['data']
        sh = f.shape
        line = f[:8,100,100]
        foo = np.array(2*range(4))
        f.file.close()
        self.assertEqual(sh,(79,200,200),)
        np.testing.assert_array_equal(line,foo)

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)


if __name__ == "__main__":
    unittest.main()
