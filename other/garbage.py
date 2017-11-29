# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Demonstrates garbage messages printed to stderr for membership
    testing, when performed in new threads.
"""

from threading import Thread

import h5py

def demonstrate():
    with h5py.File('foo', 'w', driver='core') as f:
        print('x' in f)

if __name__ == '__main__':
    print("Main thread")
    demonstrate()
    thread = Thread(target=demonstrate)
    print("New thread")
    thread.start()
    thread.join()
