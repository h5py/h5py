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

include "defs.pxd"

cdef class H5PYConfig:

    cdef object _r_name
    cdef object _i_name
    cdef object _f_name
    cdef object _t_name
    cdef readonly object API_16
    cdef readonly object API_18
    cdef readonly object DEBUG
    cdef readonly object THREADS

cpdef H5PYConfig get_config()

cdef class ObjectID:

    cdef object __weakref__
    cdef readonly hid_t id
    cdef readonly int _locked
    cdef object _hash

cdef class SmartStruct:
    cdef object __weakref__
    cdef object _title

# Library init.  Safe to call more than once.
cdef int init_hdf5() except -1



