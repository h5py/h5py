Announcing HDF5 for Python (h5py) 2.2.1
=======================================

The h5py team is happy to announce the availability of h5py 2.2.1.  This
release fixes a critical bug reported by Jim Parker on December 7th, which
affects code using HDF5 compound types.

We recommend that all users of h5py upgrade to avoid crashes or possible
data corruption.


Scope of bug
------------

The issue may affect code which modifies HDF5 compound types in-place, by
specifying a field name or names when writing to a dataset:

>>> dataset['field_name'] = value

Under certain conditions, h5py can supply uninitialized memory to the HDF5
conversion machinery, leading (in the case reported) to a segmentation fault.
It is also possible for other fields of the type to be corrupted.

This issue affects only code which updates a subset of the fields in the
compound type.  Programs reading from a compound type, writing all fields, or
using other datatypes, are not affected.  If you are using code that takes
advantage of this feature, please double-check your files to ensure your data
has not been affected.


More information
----------------

Github issue:  https://github.com/h5py/h5py/issues/372
Original thread: https://groups.google.com/forum/#!topic/h5py/AbUOZ1MXf3U
