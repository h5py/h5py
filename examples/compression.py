
"""
    Example demonstrating how to use compression and other special options
    for storing datasets in HDF5.

    Compression is supported in HDF5 via a "filter pipeline" which is applied
    to data as it is written to and read from disk.  Each dataset in the
    file has its own pipeline, which allows the compression strategy to be
    specified on a per-dataset basis.

    Compression is only available for the actual data, and not for attributes
    or metadata.

    As of h5py 1.1, three compression techniques are available, "gzip", "lzf",
    and "szip".  The non-compression filters "shuffle" and "fletcher32" are
    also available.  See the docstring for the module h5py.filters for more
    information.

    Please note LZF is a h5py-only filter.  While reference C source is
    available, other HDF5-aware applications may be unable to read data in
    this format.
"""

import os

import numpy as np
import h5py
import h5py.filters

SHAPE = (100,100,100,20)
DTYPE = np.dtype('i')
SIZE = np.product(SHAPE)

f = h5py.File('compress_test.hdf5','w')

mydata = np.arange(SIZE,dtype=DTYPE).reshape(SHAPE)

datasets = []

print "Creating dataset with gzip"
dset = f.create_dataset("gzipped", data=mydata, compression="gzip",
                         compression_opts=4)   # compression_opts is optional
datasets.append(dset)

print "Creating dataset with LZF"
dset = f.create_dataset("lzfcompressed", data=mydata, compression="lzf")
datasets.append(dset)

if 'szip' in h5py.filters.encode:       # Not distributed with all versions of HDF5
    print "Creating dataset with SZIP"
    dset = f.create_dataset("szipped", data=mydata, compression="szip",
                             compression_opts=('nn',8))
    datasets.append(dset)

print "Creating dataset with LZF and error detection"
dset = f.create_dataset("gzip_error_detection", data=mydata,
                        compression="gzip", fletcher32=True)
datasets.append(dset)

print "Creating uncompressed dataset"
dset = f.create_dataset("uncompressed", data=mydata)
datasets.append(dset)

f.flush()

def showsettings(dataset):
    """ Demonstrate the public attributes of datasets """

    print "="*60
    print "Dataset      ", dataset.name
    print '-'*30
    print "Shape        ", dataset.shape
    print "Chunk size   ", dataset.chunks
    print "Datatype     ", dataset.dtype
    print '-'*30
    print "Compression  ", dataset.compression
    print "Settings     ", dataset.compression_opts
    print '-'*32
    print "Shuffle      ", dataset.shuffle
    print "Fletcher32   ", dataset.fletcher32

for x in datasets:
    showsettings(x)

f.close()



