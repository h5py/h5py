
"""
    HDF5 for Python (h5py) is a Python interface to the HDF5 library.  Built
    on a near-complete Python wrapping of the HDF5 C API, it exposes a simple,
    NumPy-like interface for interacting with HDF5 files, datasets, attributes
    and groups.

    This is a simple module which demonstrates some of the features of HDF5,
    including the ability to interact with large on-disk datasets in a
    NumPy-like fashion.

    In this example, we create a file containing a 1 GB dataset, populate it
    from NumPy, and then slice into it.  HDF5 attributes are also demonstrated.

    HDF5 for Python is available at h5py.alfven.org.
"""

import numpy as np
import h5py

f = h5py.File('myfile.hdf5','w')

# Create a new, empty dataset to hold 1GB of floats
dset = f.create_dataset('MyDataset', (256, 1024, 1024), dtype='f')

# Datasets have some of the same properties as NumPy arrays
print "The new dataset has shape %s and type %s" % (dset.shape, dset.dtype)

# Attach some attributes
dset.attrs['purpose'] = "Demonstration dataset for floats"
dset.attrs['original size'] = (256, 1024, 1024)  # This tuple is auto-
                                                 # converted to an HDF5 array.
dset.attrs['constant'] = 42

# Populate the file in a loop.  Note that you can use NumPy-style slicing
# on datasets directly, including the row-like selection demonstrated here.

base = np.arange(1024*1024, dtype='f').reshape((1024,1024))
for idx in xrange(256):
    if(idx%16==0): print 'Populating row %d' % idx

    base += idx*(1024*1024)
    dset[idx] = base


# Perform some operations requiring random access.  Note these operations use
# HDF5 "dataspaces" for efficient read/write.

print "Resetting some indices to one"
dset[15, 24, 100:200] = np.ones((100,), dtype='f')

print 'Retrieving every 64th element... '
subarray = dset[...,::64]
print 'Retrived array has shape %s' % (subarray.shape,)

# We can also access attributes using dictionary-style syntax
for name, value in dset.attrs.iteritems():
    print 'Attribute "%s" has value: %r' % (name, value)

# When finished, close the file.  The dataset (and all other open objects)
# are closed automatically.
f.close()



