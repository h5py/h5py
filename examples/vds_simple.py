'''A simple example of building a virtual dataset.

This makes four 'source' HDF5 files, each with a 1D dataset of 100 numbers.
Then it makes a single 4x100 virtual dataset in a separate file, exposing
the four sources as one dataset.
'''

import h5py
import numpy as np

# Create source files (1.h5 to 4.h5)
for n in range(1, 5):
    with h5py.File('{}.h5'.format(n), 'w') as f:
        d = f.create_dataset('data', (100,), 'i4')
        d[:] = np.arange(100) + n

# Assemble virtual dataset
target = h5py.VirtualTarget(shape=(4, 100), dtype='i4', fillvalue=-5)

for n in range(1, 5):
    filename = "{}.h5".format(n)
    vsource = h5py.VirtualSource(filename, 'data', shape=(100,))
    target[n - 1] = vsource

# Add virtual dataset to output file
with h5py.File("VDS.h5", 'w', libver='latest') as f:
    f.create_virtual_dataset('data', target)
    print("Virtual dataset:")
    print(f['data'][:, :10])
