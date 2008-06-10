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

from defs_c cimport size_t
from h5 cimport hid_t, hsize_t
from numpy cimport ndarray

cdef extern from "utils_low.h":

    # Python to HDF5 complex conversion
    hid_t create_ieee_complex64(char byteorder, char* real_name, char* img_name)
    hid_t create_ieee_complex128(char byteorder, char* real_name, char* img_name)

    # Tuple conversion
    int convert_tuple(object tpl, hsize_t *dims, hsize_t rank) except -1
    object convert_dims(hsize_t* dims, hsize_t rank)

    # Numpy array validation
    int check_numpy_read(ndarray arr, hid_t space_id)
    int check_numpy_write(ndarray arr, hid_t space_id)

    # Memory handling
    void* emalloc(size_t size) except? NULL
    void efree(void* ptr)

# === Custom API ==============================================================

cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1


