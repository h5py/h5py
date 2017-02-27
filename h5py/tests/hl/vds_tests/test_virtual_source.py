import unittest
import h5py as h5
import numpy as np

class TestVirtualSource(unittest.TestCase):
    def test_full_slice(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[:,:,:]
        self.assertEqual(dataset.shape,sliced.shape)
    
    def test_full_slice_inverted(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[:,:,::-1]
        self.assertEqual(dataset.shape,sliced.shape)
        
    def test_subsampled_slice_inverted(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[:,:,::-2]
        self.assertEqual((20,30,15),sliced.shape)
    
    def test_integer_indexed(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5,:,:]
        self.assertEqual((1,30,30),sliced.shape)

    def test_integer_single_indexed(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5]
        self.assertEqual((1,30,30),sliced.shape)
    
    def test_two_integer_indexed(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5,:,10]
        self.assertEqual((1,30,1),sliced.shape)
    
    def test_single_range(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5:10,:,:]
        self.assertEqual((5,)+dataset.shape[1:],sliced.shape)
    
    def test_shape_calculation_positive_step(self):
        dataset = h5.VirtualSource('test','test',(20,))
        cmp = []
        for i in range(5):
            d = dataset[2:12+i:3].shape[0]
            ref = np.arange(20)[2:12+i:3].size
            cmp.append(ref==d)
        self.assertEqual(5, sum(cmp))

    def test_shape_calculation_positive_step_switched_start_stop(self):
        dataset = h5.VirtualSource('test','test',(20,))
        cmp = []
        for i in range(5):
            d = dataset[12+i:2:3].shape[0]
            ref = np.arange(20)[12+i:2:3].size
            print d,ref
            cmp.append(ref==d)
        self.assertEqual(5, sum(cmp))


    def test_shape_calculation_negative_step(self):
        dataset = h5.VirtualSource('test','test',(20,))
        cmp = []
        for i in range(5):
            d = dataset[12+i:2:-3].shape[0]
            ref = np.arange(20)[12+i:2:-3].size
            cmp.append(ref==d) 
        self.assertEqual(5, sum(cmp))
        
    def test_shape_calculation_negative_step_switched_start_stop(self):
        dataset = h5.VirtualSource('test','test',(20,))
        cmp = []
        for i in range(5):
            d = dataset[2:12+i:-3].shape[0]
            ref = np.arange(20)[2:12+i:-3].size
            cmp.append(ref==d)
        self.assertEqual(5, sum(cmp))


    def test_double_range(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5:10,:,20:25]
        self.assertEqual((5,30,5),sliced.shape)

    def test_double_strided_range(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[6:12:2,:,20:26:3]
        self.assertEqual((3,30,2,),sliced.shape)

    def test_double_strided_range_inverted(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[12:6:-2,:,26:20:-3]
        self.assertEqual((3,30,2),sliced.shape)

    def test_negative_start_index(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[-10:16]
        self.assertEqual((6,30,30),sliced.shape)
        
    def test_negative_stop_index(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[10:-4]
        self.assertEqual((6,30,30),sliced.shape)

    def test_negative_start_and_stop_index(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[-10:-4]
        self.assertEqual((6,30,30),sliced.shape)
        
    def test_negative_start_and_stop_and_stride_index(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[-4:-10:-2]
        self.assertEqual((3,30,30),sliced.shape)
#         
    def test_ellipsis(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[...]
        self.assertEqual(dataset.shape,sliced.shape)
   
    def test_ellipsis_end(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[0,...]
        self.assertEqual((1,)+dataset.shape[1:],sliced.shape)
  
    def test_ellipsis_start(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[...,0]
        self.assertEqual(dataset.shape[:-1]+(1,),sliced.shape)
  
    def test_ellipsis_sandwich(self):
        dataset = h5.VirtualSource('test','test',(20,30,30,40))
        sliced = dataset[0,...,5]
        self.assertEqual((1,)+dataset.shape[1:-1]+(1,),sliced.shape)

        

if __name__ == "__main__":
    unittest.main()
