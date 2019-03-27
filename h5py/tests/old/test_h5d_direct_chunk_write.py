from __future__ import absolute_import

import h5py
import numpy

from ..common import ut, TestCase


@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 8, 11), 'Direct Chunk Writing requires HDF5 >= 1.8.11')
class TestWriteDirectChunk(TestCase):
    def test_write_direct_chunk(self):

        filename = self.mktemp().encode()
        filehandle = h5py.File(filename, "w")

        dataset = filehandle.create_dataset("data", (100, 100, 100),
                                            maxshape=(None, 100, 100),
                                            chunks=(1, 100, 100),
                                            dtype='float32')

        # writing
        array = numpy.zeros((10, 100, 100))
        for index in range(10):
            a = numpy.random.rand(100, 100).astype('float32')
            dataset.id.write_direct_chunk((index, 0, 0), a.tostring(), filter_mask=1)
            array[index] = a

        filehandle.close()

        # checking
        filehandle = h5py.File(filename, "r")
        for i in range(10):
            read_data = filehandle["data"][i]
            self.assertTrue((array[i] == read_data).all())


@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 2), 'Direct Chunk Reading requires HDF5 >= 1.10.2')
class TestReadDirectChunk(TestCase):
    def test_read_offsets(self):

        filename = self.mktemp().encode()
        filehandle = h5py.File(filename, "w")

        slice = numpy.arange(16).reshape(4, 4)
        slice_dataset = filehandle.create_dataset("slice",
                                          data=slice,
                                          compression="gzip",
                                          compression_opts=9)
        dataset = filehandle.create_dataset("compressed_chunked",
                                            data=[slice, slice, slice],
                                            compression="gzip",
                                            compression_opts=9,
                                            chunks=(1, ) + slice.shape)
        compressed_slice, _filter_mask = slice_dataset.id.read_direct_chunk((0, 0))
        for i in range(dataset.shape[0]):
            data, _filter_mask = dataset.id.read_direct_chunk((i, 0, 0))
            self.assertEqual(compressed_slice, data)

    def test_read_write_chunk(self):

        filename = self.mktemp().encode()
        filehandle = h5py.File(filename, "w")

        # create a reference
        slice = numpy.arange(16).reshape(4, 4)
        slice_dataset = filehandle.create_dataset("source",
                                          data=slice,
                                          compression="gzip",
                                          compression_opts=9)
        # configure an empty dataset
        compressed_slice, filter_mask = slice_dataset.id.read_direct_chunk((0, 0))
        dataset = filehandle.create_dataset("created",
                                            shape=slice_dataset.shape,
                                            maxshape=slice_dataset.shape,
                                            chunks=slice_dataset.chunks,
                                            dtype=slice_dataset.dtype,
                                            compression="gzip",
                                            compression_opts=9)

        # copy the data
        dataset.id.write_direct_chunk((0, 0), compressed_slice, filter_mask=filter_mask)
        filehandle.close()

        # checking
        filehandle = h5py.File(filename, "r")
        dataset = filehandle["created"]
        self.assertTrue((dataset[...] == slice).all())
