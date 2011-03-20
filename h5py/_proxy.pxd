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

from defs cimport *

cdef herr_t attr_rw(hid_t attr, hid_t mtype, void *progbuf, int read) except -1

cdef herr_t dset_rw(hid_t dset, hid_t mtype, hid_t mspace, hid_t fspace,
                    hid_t dxpl, void* progbuf, int read) except -1

