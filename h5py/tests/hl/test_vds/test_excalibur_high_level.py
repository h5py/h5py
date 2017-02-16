'''
Unit test for the high level vds interface for excalibur
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


import unittest
import numpy as np
import h5py as h5
import tempfile

class ExcaliburData(object):
    FEM_PIXELS_PER_CHIP_X = 256
    FEM_PIXELS_PER_CHIP_Y = 256
    FEM_CHIPS_PER_STRIPE_X = 8
    FEM_CHIPS_PER_STRIPE_Y = 1
    FEM_STRIPES_PER_MODULE = 2

    @property
    def sensor_module_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y * self.FEM_STRIPES_PER_MODULE
        return y_pixels, x_pixels,

    @property
    def fem_stripe_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y
        return y_pixels, x_pixels,

    def generate_sensor_module_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.sensor_module_dimensions, dtype=dtype)
        dset.fill(value)
        return dset

    def generate_fem_stripe_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.fem_stripe_dimensions, dtype=dtype)
        dset.fill(value)
        return dset



class TestExcaliburHighLevel(unittest.TestCase):
    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data,scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = shape#(None,) + excalibur_data.fem_stripe_dimensions
        chunk = (1,) + excalibur_data.fem_stripe_dimensions
        with h5.File(fname, 'w', libver='latest') as f:
            dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
            for data_value_index in np.arange(nframes):
                dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index*scale)

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ["stripe_%d.h5" % stripe for stripe in range(1,7)]
        self.fname = [self.working_dir+ix for ix in self.fname]
        nframes = 5
        self.edata = ExcaliburData()
        k=0
        for raw_file in self.fname:
            self.create_excalibur_fem_stripe_datafile(raw_file, nframes, self.edata,k)
            k+=1

    def test_excalibur_high_level(self):
        self.outfile = self.working_dir + 'excalibur.h5'
        f = h5.File(self.outfile,'w',libver='latest') # create an output file.
        in_key = 'data' # where is the data at the input?
        in_sh = h5.File(self.fname[0],'r')[in_key].shape # get the input shape
        dtype = h5.File(self.fname[0],'r')[in_key].dtype # get the datatype
        # now generate the output shape
        vertical_gap = 10 # pixels spacing in the vertical
        nfiles = len(self.fname)
        nframes = in_sh[0]
        width = in_sh[2]
        height = (in_sh[1]*nfiles) + (vertical_gap*(nfiles-1))
        out_sh = (nframes, height, width)
        TGT = h5.VirtualTarget(self.outfile, 'data', shape=out_sh) # Virtual target is a representation of the output dataset
        offset = 0 # initial offset
        VMlist = [] # place to put the maps
        for i in range(nfiles):
            VSRC = h5.VirtualSource(self.fname[i], in_key,shape=in_sh) #a representation of the input dataset
            VM = h5.VirtualMap(VSRC, TGT[:,offset:(offset+in_sh[1]),:], dtype=dtype) # map them with indexing
            offset += in_sh[1]+vertical_gap # increment the offset
            VMlist.append(VM) # append it to the list

        f.create_virtual_dataset(VMlist=VMlist, fillvalue=0x1) # pass the fill value and list of maps
        f.close()

        f = h5.File(self.outfile,'r')['data']
        self.assertEqual(f[3,100,0], 0.0)
        self.assertEqual(f[3,260,0], 1.0)
        self.assertEqual(f[3,350,0], 3.0)
        self.assertEqual(f[3,650,0], 6.0)
        self.assertEqual(f[3,900,0], 9.0)
        self.assertEqual(f[3,1150,0], 12.0)
        self.assertEqual(f[3,1450,0], 15.0)
        f.file.close()

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)


if __name__ == "__main__":
    unittest.main()
