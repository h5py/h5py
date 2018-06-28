'''Concatenate multiple files into a single virtual dataset
'''
import h5py
import numpy as np

file_names_to_concatenate = ['1.h5', '2.h5', '3.h5', '4.h5', '5.h5']
entry_key = 'data' # where the data is inside of the source files.
sh = h5py.File(file_names_to_concatenate[0], 'r')[entry_key].shape # get the first ones shape.

layout = h5py.VirtualLayout(shape=(len(file_names_to_concatenate),) + sh,
                            dtype=np.float)

for i, filename in enumerate(file_names_to_concatenate):
    vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
    layout[i, :, :, :] = vsource

with h5py.File("VDS.h5", 'w', libver='latest') as f:
    f.create_virtual_dataset(entry_key, layout, fillvalue=0)
