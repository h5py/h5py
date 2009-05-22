
"""
    HDF5 for Python (h5py) is a Python interface to the HDF5 library.  Built
    on a near-complete Python wrapping of the HDF5 C API, it exposes a simple,
    NumPy-like interface for interacting with HDF5 files, datasets, attributes
    and groups.

    This module demonstrates the use of HDF5 groups from h5py.  HDF5 groups
    are analagous to directories in a filesystem; they even use the UNIX-style
    /path/to/resource syntax.  In h5py, groups act like dictionaries.  They
    also provide the necessary methods to create subgroups, datasets, and
    attributes.

    HDF5 for Python is available at h5py.alfven.org.
"""

import numpy as np
import h5py

f = h5py.File('myfile.hdf5','w')

# The file object is also the "root group" ("/") in HDF5.  It's currently
# empty:
print "Number of items in the root group: %d" % len(f)

# Create some groups
g1 = f.create_group('Group1')
g2 = f.create_group('Another Group')
g3 = f.create_group('Yet another group')

# All groups, including the root group, support a basic dictionary-style
# interface
print "There are now %d items in the root group" % len(f)
print "They are: %s" % ", ".join(f)  # iterating yields member names

# Groups can contain subgroups themselves
sub1 = g1.create_group("Subgroup1")

# Prints "/Group1/Subgroup1"
print "Full name of subgroup is %s" % sub1.name

# You can retrieve them using __getitem__ syntax
sub2 = g1['Subgroup1']

# You can attach attributes to groups, just like datasets, containing just
# about anything NumPy can handle.
g1.attrs['purpose'] = "A demonstration group"
g1.attrs['Life, universe, everything'] = 42
g1.attrs['A numpy array'] = np.ones((3,), dtype='>i2')

# Create datasets using group methods.  (See other examples for a more in-
# depth introduction to datasets).

data = np.arange(100*100).reshape((100,100))

dset = sub1.create_dataset("My dataset", data=data)

print "The new dataset has full name %s, shape %s and type %s" % \
        (dset.name, dset.shape, dset.dtype) 

# Closing the file closes all open objects
f.close()






