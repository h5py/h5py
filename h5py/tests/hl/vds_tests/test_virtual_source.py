import unittest
import h5py as h5


class TestVirtualSource(unittest.TestCase):
    def test_full_slice(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[:,:,:]
        self.assertEqual(dataset.shape,sliced.shape)
    
    def test_full_slice_inverted(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[:,:,::-1]
        self.assertEqual(dataset.shape,sliced.shape)
    
    def test_integer_indexed(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5,:,:]
        self.assertEqual((1,30,30),sliced.shape)
    
    def test_two_integer_indexed(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5,:,10]
        self.assertEqual((1,30,1),sliced.shape)
    
    def test_single_range(self):
        dataset = h5.VirtualSource('test','test',(20,30,30))
        sliced = dataset[5:10,:,:]
        self.assertEqual((5,)+dataset.shape[1:],sliced.shape)
        
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

if __name__ == "__main__":
    unittest.main()
