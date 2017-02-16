'''
Unit test for the low level vds interface for excalibur
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



class TestExcaliburLowLevel(unittest.TestCase):
    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data,scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = (nframes,) + excalibur_data.fem_stripe_dimensions
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

    def test_excalibur_low_level(self):

        excalibur_data = self.edata
        self.outfile = self.working_dir+'excalibur.h5'
        vdset_stripe_shape = (1,) + excalibur_data.fem_stripe_dimensions
        vdset_stripe_max_shape = (5, ) + excalibur_data.fem_stripe_dimensions
        vdset_shape = (5,
                       excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + (10 * (len(self.fname)-1)),
                       excalibur_data.fem_stripe_dimensions[1])
        vdset_max_shape = (5,
                           excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + (10 * (len(self.fname)-1)),
                           excalibur_data.fem_stripe_dimensions[1])
        vdset_y_offset = 0

        # Create the virtual dataset file
        with h5.File(self.outfile, 'w', libver='latest') as f:

            # Create the source dataset dataspace
            src_dspace = h5.h5s.create_simple(vdset_stripe_shape, vdset_stripe_max_shape)
            # Create the virtual dataset dataspace
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)

            # Create the virtual dataset property list
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([0x01]))

            # Select the source dataset hyperslab
            src_dspace.select_hyperslab(start=(0, 0, 0), count=(1, 1, 1), block=vdset_stripe_max_shape)

            for raw_file in self.fname:
                # Select the virtual dataset hyperslab (for the source dataset)
                virt_dspace.select_hyperslab(start=(0, vdset_y_offset, 0),
                                             count=(1, 1, 1),
                                             block=vdset_stripe_max_shape)
                # Set the virtual dataset hyperslab to point to the real first dataset
                dcpl.set_virtual(virt_dspace, raw_file, "/data", src_dspace)
                vdset_y_offset += vdset_stripe_shape[1] + 10

            # Create the virtual dataset
            dset = h5.h5d.create(f.id, name="data", tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)
            assert(f['data'].fillvalue == 0x01)

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
