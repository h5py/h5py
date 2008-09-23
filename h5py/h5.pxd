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

cdef class PHIL:

    cdef object lock

    cpdef bint __enter__(self) except -1
    cpdef bint __exit__(self, a, b, c) except -1
    cpdef bint acquire(self, int blocking=*) except -1
    cpdef bint release(self) except -1

cpdef PHIL get_phil()

cdef class H5PYConfig:

    cdef object _complex_names
    cdef readonly object API_16
    cdef readonly object API_18
    cdef readonly object DEBUG
    cdef readonly object THREADS

cdef class ObjectID:

    cdef object __weakref__
    cdef readonly hid_t id
    cdef readonly int _locked
    cdef object _hash

cdef object standard_richcmp(object self, object other, int how)
cdef object obj_hash(ObjectID obj)
cdef int init_hdf5() except -1



