# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Demonstrates memory leak involving variable-length strings.
"""

import sys
import resource
import numpy as np

import h5py

FNAME = 'test.hdf5'

if 'linux' in sys.platform:
    MAXRSS_BYTES = 1024. # in KiB on linux
else:
    MAXRSS_BYTES = 1.

memory = 0
def print_memory():
    global memory

    rubytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*MAXRSS_BYTES
    print ("%.2f MB (%.2f since last call)" % (rubytes/(1024.**2), (rubytes-memory)/(1024.**2)))
    memory = rubytes


def make_data(kind):
    global data
    global dt

    if kind is bytes:
        s = b"xx"
    else:
        s = b"xx".decode('utf8')

    dt = h5py.vlen_dtype(kind)
    data = np.array([s*100 for idx in range(1000)])


def ds_leak():
    print("Testing vlens for dataset r/w")
    print("-----------------------------")
    with h5py.File(FNAME,'w') as f:
        ds = f.create_dataset('dset', (1000,), dtype=dt)
        for idx in range(500):
            #print idx
            if idx%100 == 0:
                print_memory()
            ds[...] = data
            ds[...]


def attr_leak():
    print("Testing vlens for attribute r/w")
    print("-------------------------------")
    with h5py.File(FNAME,'w') as f:
        for idx in range(500):
            if idx%100 == 0:
                print_memory()
            f.attrs.create('foo', dtype=dt, data=data)
            f.attrs['foo']


if __name__ == '__main__':
    print("h5py ", h5py.version.version)
    print("HDF5 ", h5py.version.hdf5_version)
    print("Bytes test")
    print("==========")
    make_data(bytes)
    attr_leak()
    ds_leak()
    print("Unicode test")
    print("============")
    make_data(unicode)
    attr_leak()
    ds_leak()
