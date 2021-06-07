"""Write compressed chunks directly, bypassing HDF5's filters
"""
import h5py
import numpy as np
import zlib

f = h5py.File("direct_chunk.h5", "w")

block_size = 2048
dataset = f.create_dataset(
    "data", (256, 1024, 1024), dtype="uint16", chunks=(64, 128, 128),
    compression="gzip", compression_opts=4,
)
# h5py's compression='gzip' is actually a misnomer: gzip does the same
# compression, but adds some extra metadata before & after the compressed data.
# This won't work if you use gzip.compress() instead of zlib!

# Random numbers with only a few possibilities, so some compression is possible.
array = np.random.randint(0, 10, size=(64, 128, 128), dtype=np.uint16)

# Compress the data, and write it into the dataset. (0, 0, 128) are coordinates
# for the start of a chunk. Equivalent to:
#   dataset[0:64, 0:128, 128:256] = array
compressed = zlib.compress(array, level=4)
dataset.id.write_direct_chunk((0, 0, 128), compressed)
print(f"Written {len(compressed)} bytes compressed data")

# Read the chunk back (HDF5 will decompress it) and check the data is the same.
read_data = dataset[:64, :128, 128:256]
np.testing.assert_array_equal(read_data, array)
print(f"Verified array of {read_data.size} elements ({read_data.nbytes} bytes)")

f.close()
