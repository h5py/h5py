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

cdef extern from "typeconv.h":

    hid_t h5py_object_type() except *
    int h5py_register_conv() except -1

cdef extern from "typeproxy.h":
    ctypedef enum h5py_rw_t:
        H5PY_WRITE = 0,
        H5PY_READ

    herr_t H5PY_dset_rw(hid_t dset, hid_t mtype, hid_t mspace_in, hid_t fspace_in,
                   hid_t xfer_plist, void* buf, h5py_rw_t dir) except *

    herr_t H5PY_attr_rw(hid_t attr, hid_t mtype, void* buf, h5py_rw_t dir) except *

cdef hid_t get_object_type() except -1

cdef herr_t attr_rw(hid_t attr_id, hid_t mem_type_id, void *buf, h5py_rw_t dir) except *

cdef herr_t dset_rw(hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, 
                  hid_t file_space_id, hid_t xfer_plist_id, void *outbuf,
                  h5py_rw_t dir) except *


