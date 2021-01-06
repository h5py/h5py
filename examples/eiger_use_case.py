'''Virtual datasets: The 'Eiger' use case

https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''

import h5py
import numpy as np


files = ['1.h5', '2.h5', '3.h5', '4.h5', '5.h5']
entry_key = 'data' # where the data is inside of the source files.
sh = h5py.File(files[0], 'r')[entry_key].shape # get the first ones shape.

layout = h5py.VirtualLayout(shape=(len(files) * sh[0], ) + sh[1:], dtype=np.float64)
M_start = 0
for i, filename in enumerate(files):
    M_end = M_start + sh[0]
    vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
    layout[M_start:M_end:1, :, :] = vsource
    M_start = M_end

with h5py.File("eiger_vds.h5", 'w', libver='latest') as f:
    f.create_virtual_dataset('data', layout, fillvalue=0)
