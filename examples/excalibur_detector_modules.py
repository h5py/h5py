'''
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
The excalibur use case
'''

import numpy as np
import h5py as h5
raw_files = ["stripe_%d.h5" % stripe for stripe in range(1,7)]# get these names
outfile = 'full_detector.h5'
f = h5.File(outfile,'w',libver='latest') # create an output file.
in_key = 'data' # where is the data at the input?
in_sh = h5.File(raw_files[0],'r')[in_key].shape # get the input shape
dtype = h5.File(raw_files[0],'r')[in_key].dtype # get the datatype

outkey = 'full_frame' # where should it go in the output file

# now generate the output shape
vertical_gap = 10 # pixels spacing in the vertical
nfiles = len(raw_files)
nframes = in_sh[0]
width = in_sh[2]
height = (in_sh[1]*nfiles) + (vertical_gap*(nfiles-1))
out_sh = (nframes, height, width)

TGT = h5.VirtualTarget(outfile, outkey, shape=out_sh) # Virtual target is a representation of the output dataset
offset = 0 # initial offset
VMlist = [] # place to put the maps
for i in range(nfiles):
    print("frame_number is: %s" % str(i)) # for feedback
    VSRC = h5.VirtualSource(raw_files[i], in_key,shape=in_sh) #a representation of the input dataset 
    VM = h5.VirtualMap(VSRC, TGT[:,offset:(offset+in_sh[1]),:], dtype=dtype) # map them with indexing
    offset += in_sh[1]+vertical_gap # increment the offset
    VMlist.append(VM) # append it to the list
    
f.create_virtual_dataset(VMlist=VMlist, fillvalue=0x1) # pass the fill value and list of maps
f.close()# close