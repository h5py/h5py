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

from numpy cimport ndarray

cdef void* emalloc(size_t size) except? NULL
cdef void efree(void* ptr)

cpdef int check_numpy_read(ndarray arr, hid_t space_id=*) except -1
cpdef int check_numpy_write(ndarray arr, hid_t space_id=*) except -1

cdef int convert_tuple(object tuple, hsize_t *dims, hsize_t rank) except -1
cdef object convert_dims(hsize_t* dims, hsize_t rank)

cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1

cdef object create_numpy_hsize(int rank, hsize_t* dims)
cdef object create_hsize_array(object arr)













