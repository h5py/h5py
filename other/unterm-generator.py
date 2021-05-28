"""Demonstrated an error garbage collecting an h5py object at Python shutdown

https://github.com/h5py/h5py/issues/1495
"""

import h5py

def yield_groups(filename):
    with h5py.File(filename, 'r') as fh:
        for group in fh:
            yield group

filename = "file_with_10_groups.hdf5"
grp_generator = yield_groups(filename)
next(grp_generator)
