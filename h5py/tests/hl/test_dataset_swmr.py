from __future__ import absolute_import

import numpy as np
import h5py

from ..common import ut, TestCase


@ut.skipUnless(h5py.version.hdf5_version_tuple < (1, 9, 178), 'SWMR is available. Skipping backwards compatible tests')
class TestDatasetNoSwmrRead(TestCase):
    """ Test backwards compatibility behaviour when using SWMR functions with 
    an older version of HDF5 which does not have this feature available.
    Skip this test if SWMR features *are* available in the HDF5 library.
    """
    
    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)
        fname = self.f.filename
        self.f.close()
       
        self.f = h5py.File(fname, 'r', swmr=True)
        self.dset = self.f['data']
        
    def test_read_data(self):
        self.assertArrayEqual(self.dset, self.data)
        
    def test_refresh_raises(self):
        with self.assertRaises(AttributeError):
            self.dset.refresh()
            
@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 9, 178), 'SWMR is available when HDF5 >= 1.9.178')
class TestDatasetSwmrRead(TestCase):
    """ sing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """
    
    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)
        fname = self.f.filename
        self.f.close()
       
        self.f = h5py.File(fname, 'r', swmr=True)
        self.dset = self.f['data']
        
    def test_read_data(self):
        self.assertArrayEqual(self.dset, self.data)
        
    def test_refresh(self):
        self.dset.refresh()
        
            

