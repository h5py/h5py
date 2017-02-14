'''
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
The excalibur use case
'''
import h5py as h5
f = h5.File('full_time_series.h5','w',libver='latest') # create an output file.
in_key = 'data' # where is the data at the input?
dtype = h5.File('raw_file_1.h5')['data'].dtype
outshape = (799,2000,2000)
TGT = h5.VirtualTarget('full_time_series.h5', in_key, shape=outshape) # Virtual target is a representation of the output dataset
VSRC1 = h5.VirtualSource('raw_file_1.h5', 'data',shape=(200,2000,2000)) #a representation of the input dataset
VSRC2 = h5.VirtualSource('raw_file_2.h5', 'data',shape=(200,2000,2000)) #a representation of the input dataset 
VSRC3 = h5.VirtualSource('raw_file_3.h5', 'data',shape=(200,2000,2000)) #a representation of the input dataset 
VSRC4 = h5.VirtualSource('raw_file_4.h5', 'data',shape=(199,2000,2000)) #a representation of the input dataset 
a = TGT[0:799:4,:,:]
b = TGT[1:799:4,:,:]
c = TGT[2:799:4,:,:]
d = TGT[3:799:4,:,:]

VM1 = h5.VirtualMap(VSRC1,a , dtype=dtype) # map them with indexing
VM2 = h5.VirtualMap(VSRC2,b , dtype=dtype) # map them with indexing
VM3 = h5.VirtualMap(VSRC3,c , dtype=dtype) # map them with indexing
VM4 = h5.VirtualMap(VSRC4,d , dtype=dtype) # map them with indexing

f.create_virtual_dataset(VMlist=[VM1,VM2,VM3,VM4], fillvalue=0x1) # pass the fill value and list of maps
f.close()# close