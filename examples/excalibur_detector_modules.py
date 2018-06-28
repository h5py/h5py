'''Virtual datasets: The 'Excalibur' use case

https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''

import h5py

raw_files = ["stripe_%d.h5" % stripe for stripe in range(1,7)]# get these names
in_key = 'data' # where is the data at the input?
outfile = 'full_detector.h5'
out_key = 'full_frame'

in_sh = h5py.File(raw_files[0], 'r')[in_key].shape # get the input shape
dtype = h5py.File(raw_files[0], 'r')[in_key].dtype # get the datatype

# now generate the output shape
vertical_gap = 10 # pixels spacing in the vertical
nfiles = len(raw_files)
nframes = in_sh[0]
width = in_sh[2]
height = (in_sh[1]*nfiles) + (vertical_gap*(nfiles-1))
out_sh = (nframes, height, width)

# Virtual target is a representation of the output dataset
layout = h5py.VirtualLayout(shape=out_sh, dtype=dtype)
offset = 0 # initial offset
for i in range(nfiles):
    print("frame_number is: %s" % str(i)) # for feedback
    vsource = h5py.VirtualSource(raw_files[i], in_key, shape=in_sh) #a representation of the input dataset
    layout[:, offset:(offset + in_sh[1]), :] = vsource
    offset += in_sh[1]+vertical_gap # increment the offset

# Create an output file.
with h5py.File(outfile, 'w', libver='latest') as f:
    f.create_virtual_dataset(out_key, layout, fillvalue=0x1)
