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

from python cimport PyTuple_Check, PyList_Check, PyErr_SetString, Py_INCREF
from numpy cimport import_array, NPY_UINT16, NPY_UINT32, NPY_UINT64, \
                   npy_intp, PyArray_SimpleNew, PyArray_ContiguousFromAny, \
                    PyArray_FROM_OTF, NPY_CONTIGUOUS, NPY_NOTSWAPPED, \
                    NPY_FORCECAST

import_array()

cdef int convert_tuple(object tpl, hsize_t *dims, hsize_t rank) except -1:
    # Convert a Python tuple to an hsize_t array.  You must allocate
    # the array yourself and pass both it and the size to this function.
    # Returns 0 on success, -1 on failure and raises an exception.
    cdef int i

    if len(tpl) != rank:
        raise ValueError("Tuple length incompatible with array")
    
    try:
        for i from 0<=i<rank:
            dims[i] = tpl[i]
    except TypeError:
        raise TypeError("Can't convert element %d (%s) to hsize_t" % (i, tpl[i]))

    return 0
    
cdef object convert_dims(hsize_t* dims, hsize_t rank):
    # Convert an hsize_t array to a Python tuple of long ints.

    cdef list dims_list
    cdef int i
    dims_list = []

    for i from 0<=i<rank:
        dims_list.append(dims[i])

    return tuple(dims_list)
    

cdef object create_numpy_hsize(int rank, hsize_t* dims):
    # Create an empty Numpy array which can hold HDF5 hsize_t entries

    cdef int typecode
    cdef npy_intp* dims_npy
    cdef ndarray arr
    cdef int i

    if sizeof(hsize_t) == 2:
        typecode = NPY_UINT16
    elif sizeof(hsize_t) == 4:
        typecode = NPY_UINT32
    elif sizeof(hsize_t) == 8:
        typecode = NPY_UINT64
    else:
        raise RuntimeError("Can't map hsize_t %d to Numpy typecode" % sizeof(hsize_t))

    dims_npy = <npy_intp*>emalloc(sizeof(npy_intp)*rank)

    try:
        for i from 0<=i<rank:
            dims_npy[i] = dims[i]
        arr = PyArray_SimpleNew(rank, dims_npy, typecode)
    finally:
        efree(dims_npy)

    return arr

cdef object create_hsize_array(object arr):
    # Create a NumPy array of hsize_t uints initialized to an existing array

    cdef int typecode
    cdef ndarray outarr

    if sizeof(hsize_t) == 2:
        typecode = NPY_UINT16
    elif sizeof(hsize_t) == 4:
        typecode = NPY_UINT32
    elif sizeof(hsize_t) == 8:
        typecode = NPY_UINT64
    else:
        raise RuntimeError("Can't map hsize_t %d to Numpy typecode" % sizeof(hsize_t))

    return PyArray_FROM_OTF(arr, typecode, NPY_CONTIGUOUS | NPY_NOTSWAPPED | NPY_FORCECAST)
    
cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1:
    # Ensure that tpl is in fact a tuple, or None if none_allowed is nonzero.
    # If size >= 0, also ensure that the length matches.
    # Otherwise raises ValueError

    if (tpl is None and none_allowed) or \
      ( PyTuple_Check(tpl) and (size < 0 or len(tpl) == size)):
        return 1

    nmsg = ""
    smsg = ""
    if size >= 0:
        smsg = " of size %d" % size
    if none_allowed:
        nmsg = " or None"

    msg = "%s must be a tuple%s%s." % (name, smsg, nmsg)
    PyErr_SetString(ValueError, msg)
    return -1

cdef int require_list(object lst, int none_allowed, int size, char* name) except -1:
    # Counterpart of require_tuple, for lists

    if (lst is None and none_allowed) or \
      (PyList_Check(lst) and (size < 0 or len(lst) == size)):
        return 1

    nmsg = ""
    smsg = ""
    if size >= 0:
        smsg = " of size %d" % size
    if none_allowed:
        nmsg = " or None"

    msg = "%s must be a list%s%s." % (name, smsg, nmsg)
    PyErr_SetString(ValueError, msg)
    return -1

cdef object pybool(long long val):
    # It seems Pyrex's bool() actually returns some sort of int.
    # This is OK for C, but ugly in Python.
    if val:
        return True
    return False



