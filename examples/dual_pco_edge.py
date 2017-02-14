'''
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
The dual pco edge use case
'''

import h5py as h5

f = h5.File('outfile.h5','w',libver='latest') # create an output file.
in_sh = h5.File('raw_file_1.h5','r')['data'].shape # get the input shape
dtype = h5.File('raw_file_1.h5','r')['data'].dtype # get the datatype
VSRC1 = h5.VirtualSource('raw_file_1.h5', 'data',shape=in_sh) #a representation of the input dataset
VSRC2 = h5.VirtualSource('raw_file_2.h5', 'data',shape=in_sh) #a representation of the input dataset 
TGT = h5.VirtualTarget('outfile.h5', 'data', shape=(in_sh[0], 2*in_sh[1]+gap, in_sh[3]))
VM1 = h5.VirtualMap(VSRC1, TGT[0:in_sh[0]:1,:,:], dtype=dtype)
VM2 = h5.VirtualMap(VSRC2, TGT[(in_sh[0]+gap):(2*in_sh[0]+gap+1):1,:,:], dtype=dtype)
f.create_virtual_dataset(VMlist=[VM1, VM2], fillvalue=0x1) # pass the fill value and list of maps
f.close()# close

