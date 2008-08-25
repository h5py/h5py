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

include "std_defs.pxi"

from numpy cimport ndarray

cdef extern from "utils_low.h":

    # Python to HDF5 complex conversion
    hid_t create_ieee_complex64(char byteorder, char* real_name, char* img_name) except -1
    hid_t create_ieee_complex128(char byteorder, char* real_name, char* img_name) except -1

    # Tuple conversion
    object convert_dims(hsize_t* dims, hsize_t rank) # automatic except

    # Numpy array validation
    int check_numpy_read(ndarray arr, hid_t space_id) except 0
    int check_numpy_write(ndarray arr, hid_t space_id) except 0

    # Memory handling
    void* emalloc(size_t size) except? NULL
    void efree(void* ptr)

# === Custom API ==============================================================

cdef int convert_tuple(object tuple, hsize_t *dims, hsize_t rank) except -1
cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1
cdef int require_list(object lst, int none_allowed, int size, char* name) except -1
cdef object pybool(long long val)
cdef object create_numpy_hsize(int rank, hsize_t* dims)
cdef object create_hsize_array(object arr)

