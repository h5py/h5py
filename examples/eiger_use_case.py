'''
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
The eiger use case
'''

import h5py as h5
import numpy as np

f = h5py.File("VDS.h5", 'w', libver='latest')
files = ['1.h5', '2.h5', '3.h5', '4.h5', '5.h5']
entry_key = 'data' # where the data is inside of the source files.
sh = h5.File(file_names_to_concatenate[0],'r')[entry_key].shape # get the first ones shape.

TGT = h5.VirtualTarget(outfile, outkey, shape=(len(file_names_to_concatenate, ) + sh))
M_minus_1 = 0
for i in range(len(files)):
    M = M_minus_1 +sh[0]
    VSRC = h5.VirtualSource(file_names_to_concatenate[i], entry_key, shape=sh)
    VM = h5.VirtualMap(VSRC, TGT[M_minus_1:M:1,:,:],dtype=np.float)
    VMlist.append(VM)
    M_minus_1 = M

d = f.create_virtual_dataset(VMlist=VMlist,fillvalue=0)
f.close()
# 


