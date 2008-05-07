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

from h5 cimport hid_t, hsize_t
from numpy cimport ndarray

cdef extern from "utils.h":

    hid_t create_ieee_complex64(char byteorder, char* real_name, char* img_name)
    hid_t create_ieee_complex128(char byteorder, char* real_name, char* img_name)
    hsize_t* tuple_to_dims(object tpl)
    object dims_to_tuple(hsize_t* dims, hsize_t rank)

    int check_numpy_read(ndarray arr, hid_t space_id)
    int check_numpy_write(ndarray arr, hid_t space_id)
