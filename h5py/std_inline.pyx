#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

# Simple defs which aren't worth putting in their own module.

cdef object pybool(long val):
    # It seems Pyrex's bool() actually returns some sort of int.
    if val:
        return True
    return False
