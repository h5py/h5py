import numpy
import numpy.testing

from h5py import File, version as h5py_version
from .common import ut, TestCase


@ut.skipUnless(h5py_version.hdf5_version_tuple >= (1, 8, 11), 'Direct Chunk Writing requires HDF5 >= 1.8.11')
class TestWriteDirectChunk(TestCase):
    def test_write_direct_chunk(self):
        dataset = self.f.create_dataset(
            "data", (100, 100, 100),
            maxshape=(None, 100, 100),
            chunks=(1, 100, 100),
            dtype='float32'
        )

        # writing
        array = numpy.zeros((10, 100, 100))
        for index in range(10):
            a = numpy.random.rand(100, 100).astype('float32')
            dataset.id.write_direct_chunk((index, 0, 0), a.tostring(), filter_mask=1)
            array[index] = a

        filename = self.f.filename
        self.f.close()

        # checking
        with File(filename, "r") as f:
            for i in range(10):
                read_data = f["data"][i]
                numpy.testing.assert_array_equal(array[i], read_data)


@ut.skipUnless(h5py_version.hdf5_version_tuple >= (1, 10, 2), 'Direct Chunk Reading requires HDF5 >= 1.10.2')
class TestReadDirectChunk(TestCase):
    def test_read_compressed_offsets(self):
        frame = numpy.arange(16).reshape(4, 4)
        frame_dataset = self.f.create_dataset(
            "frame", data=frame, compression="gzip", compression_opts=9
        )
        dataset = self.f.create_dataset(
            "compressed_chunked",
            data=[frame, frame, frame],
            compression="gzip",
            compression_opts=9,
            chunks=(1, ) + frame.shape,
        )
        filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0))
        # No filter must be disabled
        self.assertEqual(filter_mask, 0)

        for i in range(dataset.shape[0]):
            filter_mask, data = dataset.id.read_direct_chunk((i, 0, 0))
            self.assertEqual(compressed_frame, data)
            # No filter must be disabled
            self.assertEqual(filter_mask, 0)

    def test_read_uncompressed_offsets(self):
        frame = numpy.arange(16).reshape(4, 4)
        dataset = self.f.create_dataset(
            "frame",
            maxshape=(1,) + frame.shape,
            shape=(1,) + frame.shape,
            compression="gzip",
            compression_opts=9
        )
        # Write uncompressed data
        DISABLE_ALL_FILTERS = 0xFFFFFFFF
        dataset.id.write_direct_chunk(
            (0, 0, 0),
            frame.tostring(),
            filter_mask=DISABLE_ALL_FILTERS
        )

        filename = self.f.filename
        self.f.close()

        # FIXME: Here we have to close the file and load it back else a runtime
        # error occurs:
        #   RuntimeError: Can't get storage size of chunk (chunk storage is not allocated)
        with File(filename, "r") as f:
            dataset = f["frame"]
            filter_mask, compressed_frame = dataset.id.read_direct_chunk((0, 0, 0))

        # At least 1 filter is supposed to be disabled
        self.assertNotEqual(filter_mask, 0)
        self.assertEqual(compressed_frame, frame.tostring())

    def test_read_write_chunk(self):
        # create a reference
        frame = numpy.arange(16).reshape(4, 4)
        frame_dataset = self.f.create_dataset(
            "source", data=frame, compression="gzip", compression_opts=9
        )
        # configure an empty dataset
        filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0))
        dataset = self.f.create_dataset(
            "created",
            shape=frame_dataset.shape,
            maxshape=frame_dataset.shape,
            chunks=frame_dataset.chunks,
            dtype=frame_dataset.dtype,
            compression="gzip",
            compression_opts=9
        )

        # copy the data
        dataset.id.write_direct_chunk((0, 0), compressed_frame, filter_mask=filter_mask)

        filename = self.f.filename
        self.f.close()

        # checking
        with File(filename, "r") as f:
            dataset = f["created"][...]
            numpy.testing.assert_array_equal(dataset, frame)
