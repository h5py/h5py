"""A simple example of building a virtual dataset.

This makes four 'source' HDF5 files, each with a 1D dataset of 100 numbers.
Then it makes a single 4x100 virtual dataset in a separate file, exposing
the four sources as one dataset.
"""

import h5py
import numpy as np

# create some sample data
data = np.arange(0, 100).reshape(1, 100) + np.arange(1, 5).reshape(4, 1)

# Create source files (0.h5 to 3.h5)
for n in range(4):
    with h5py.File(f"{n}.h5", "w") as f:
        d = f.create_dataset("data", (100,), "i4", data[n])

# Assemble virtual dataset
layout = h5py.VirtualLayout(shape=(4, 100), dtype="i4")
for n in range(4):
    filename = "{}.h5".format(n)
    vsource = h5py.VirtualSource(filename, "data", shape=(100,))
    layout[n] = vsource

# Add virtual dataset to output file
with h5py.File("VDS.h5", "w", libver="latest") as f:
    f.create_virtual_dataset("vdata", layout, fillvalue=-5)
    f.create_dataset("data", data=data, dtype="i4")


# read data back
# virtual dataset is transparent for reader!
with h5py.File("VDS.h5", "r") as f:
    print("Virtual dataset:")
    print(f["vdata"][:, :10])
    print("Normal dataset:")
    print(f["data"][:, :10])
