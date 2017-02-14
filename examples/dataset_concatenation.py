'''
using the example refactored vds code
'''
import h5py as h5
import numpy as np

f = h5py.File("VDS.h5", 'w', libver='latest')
file_names_to_concatenate = ['1.h5', '2.h5', '3.h5', '4.h5', '5.h5']
entry_key = 'data' # where the data is inside of the source files.
sh = h5.File(file_names_to_concatenate[0],'r')[entry_key].shape # get the first ones shape.

TGT = h5.VirtualTarget(outfile, outkey, shape=(len(file_names_to_concatenate, ) + sh)

for i in range(num_projections):
    VSRC = h5.VirtualSource(file_names_to_concatenate[i]), entry_key, shape=sh)
    VM = h5.VirtualMap(VSRC[:,:,:], TGT[i:(i+1):1,:,:,:],dtype=np.float)
    VMlist.append(VM)

d = f.create_virtual_dataset(VMlist=VMlist,fillvalue=0)
f.close()
# 


