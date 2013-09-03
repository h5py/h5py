# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Demonstrates deadlock related to attribute iteration.
"""

from threading import Thread
import sys

import h5py

FNAME = "deadlock.hdf5"

def make_file():
    with h5py.File(FNAME,'w') as f:
        for idx in range(1000):
            f.attrs['%d'%idx] = 1

def list_attributes():
    with h5py.File(FNAME, 'r') as f:
        names = list(f.attrs)

if __name__ == '__main__':

    make_file()
    thread = Thread(target=list_attributes)
    thread.start()
    list_attributes()
    thread.join()
