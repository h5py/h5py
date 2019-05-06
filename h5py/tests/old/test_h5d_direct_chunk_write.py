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

        frame = numpy.arange(16).reshape(4, 4)
        frame_dataset = filehandle.create_dataset("frame",
                                                  data=frame,
                                                  compression="gzip",
                                                  compression_opts=9)
        dataset = filehandle.create_dataset("compressed_chunked",
                                            data=[frame, frame, frame],
                                            compression="gzip",
                                            compression_opts=9,
                                            chunks=(1, ) + frame.shape)
        _filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0), filter_mask=0xFFFF)
        for i in range(dataset.shape[0]):
            _filter_mask, data = dataset.id.read_direct_chunk((i, 0, 0), filter_mask=0xFFFF)
            self.assertEqual(compressed_frame, data)

    def test_read_write_chunk(self):

        filename = self.mktemp().encode()
        filehandle = h5py.File(filename, "w")

        # create a reference
        frame = numpy.arange(16).reshape(4, 4)
        frame_dataset = filehandle.create_dataset("source",
                                                  data=frame,
                                                  compression="gzip",
                                                  compression_opts=9)
        # configure an empty dataset
        filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0), filter_mask=0xFFFF)
        dataset = filehandle.create_dataset("created",
                                            shape=frame_dataset.shape,
                                            maxshape=frame_dataset.shape,
                                            chunks=frame_dataset.chunks,
                                            dtype=frame_dataset.dtype,
                                            compression="gzip",
                                            compression_opts=9)

        # copy the data
        dataset.id.write_direct_chunk((0, 0), compressed_frame, filter_mask=filter_mask)
        filehandle.close()

        # checking
        filehandle = h5py.File(filename, "r")
        dataset = filehandle["created"]
        self.assertTrue((dataset[...] == frame).all())
