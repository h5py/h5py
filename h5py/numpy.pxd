# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2019 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

# This file is based on code from the PyTables project.  The complete PyTables
# license is available at licenses/pytables.txt, in the distribution root
# directory.

# cImport types
from numpy cimport npy_intp, npy_int8, npy_uint8, npy_int16, npy_uint16,\
    npy_int32, npy_uint32, npy_int64, npy_uint64, npy_float32, npy_float64,\
    npy_cfloat, npy_cdouble,\
    NPY_BOOL, NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT, NPY_UINT,\
    NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG, NPY_FLOAT, NPY_DOUBLE,\
    NPY_LONGDOUBLE, NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE, NPY_OBJECT,\
    NPY_STRING, NPY_UNICODE, NPY_VOID, NPY_NTYPES, NPY_NOTYPE,\
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,\
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,\
    NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128,\
    NPY_WRITEABLE, NPY_ALIGNED, NPY_C_CONTIGUOUS, NPY_CONTIGUOUS,\
    NPY_FORCECAST, NPY_NOTSWAPPED, NPY_OWNDATA\

# cImport functions
from numpy cimport PyArray_DIM, PyArray_FROM_OF, PyArray_GETITEM, PyArray_SETITEM,\
    PyArray_DescrFromType, PyArray_NBYTES, PyArray_CheckScalar,\
    PyArray_ScalarAsCtype, PyArray_SimpleNew, PyArray_ContiguousFromAny, PyArray_FROM_OTF,\
    import_array, PyArray_DATA, ndarray

# API for NumPy objects
cdef extern from "numpy/arrayobject.h":
    object PyArray_Scalar (void *, dtype, object)

#   # Classes
    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        cdef:
            int type_num, elsize, alignment
            char type, kind, byteorder, hasobject
            object fields, typeobj
