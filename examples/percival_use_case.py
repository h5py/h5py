'''Virtual datasets: the 'Percival Frame Builder' use case.

https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''
import h5py

in_key = 'data' # where is the data at the input?
dtype = h5py.File('raw_file_1.h5')['data'].dtype
outshape = (799,2000,2000)

# Virtual target is a representation of the output dataset
layout = h5py.VirtualLayout(shape=outshape, dtype=dtype)

# Sources represent the input datasets
vsource1 = h5py.VirtualSource('raw_file_1.h5', 'data', shape=(200, 2000, 2000))
vsource2 = h5py.VirtualSource('raw_file_2.h5', 'data', shape=(200, 2000, 2000))
vsource3 = h5py.VirtualSource('raw_file_3.h5', 'data', shape=(200, 2000, 2000))
vsource4 = h5py.VirtualSource('raw_file_4.h5', 'data', shape=(199, 2000, 2000))

# Map the inputs into the virtual dataset
layout[0:799:4, :, :] = vsource1
layout[1:799:4, :, :] = vsource2
layout[2:799:4, :, :] = vsource3
layout[3:799:4, :, :] = vsource4

# Create an output file
with h5py.File('full_time_series.h5', 'w', libver='latest') as f:
    f.create_virtual_dataset(in_key, layout, fillvalue=0x1)
