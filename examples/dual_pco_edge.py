'''Virtual datasets: The 'Dual PCO Edge' use case

https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''

import h5py

with h5py.File('raw_file_1.h5', 'r') as f:
    in_sh = f['data'].shape # get the input shape
    dtype = f['data'].dtype # get the datatype

gap = 10

# Sources represent the input datasets
vsource1 = h5py.VirtualSource('raw_file_1.h5', 'data', shape=in_sh)
vsource2 = h5py.VirtualSource('raw_file_2.h5', 'data', shape=in_sh)
# target is where we layout the virtual dataset
layout = h5py.VirtualLayout((in_sh[0], 2 * in_sh[1] + gap, in_sh[3]),
                            dtype=dtype)
layout[0:in_sh[0]:1, :, :] = vsource1
layout[(in_sh[0] + gap):(2 * in_sh[0] + gap + 1):1, :, :] = vsource2

# Create an output file
with h5py.File('outfile.h5', 'w', libver='latest') as f:
    f.create_virtual_dataset('data', layout, fillvalue=0x1)
