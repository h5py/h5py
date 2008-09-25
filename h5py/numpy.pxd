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

# This file is based on code from the PyTables project.  The complete PyTables
# license is available at licenses/pytables.txt, in the distribution root
# directory.

# API for NumPy objects
cdef extern from "numpy/arrayobject.h":

  # Platform independent types
  ctypedef int npy_intp
  ctypedef signed int npy_int8
  ctypedef unsigned int npy_uint8
  ctypedef signed int npy_int16
  ctypedef unsigned int npy_uint16
  ctypedef signed int npy_int32
  ctypedef unsigned int npy_uint32
  ctypedef signed long long npy_int64
  ctypedef unsigned long long npy_uint64
  ctypedef float npy_float32
  ctypedef double npy_float64

  cdef enum NPY_TYPES:
    NPY_BOOL
    NPY_BYTE
    NPY_UBYTE
    NPY_SHORT
    NPY_USHORT
    NPY_INT
    NPY_UINT
    NPY_LONG
    NPY_ULONG
    NPY_LONGLONG
    NPY_ULONGLONG
    NPY_FLOAT
    NPY_DOUBLE
    NPY_LONGDOUBLE
    NPY_CFLOAT
    NPY_CDOUBLE
    NPY_CLONGDOUBLE
    NPY_OBJECT
    NPY_STRING
    NPY_UNICODE
    NPY_VOID
    NPY_NTYPES
    NPY_NOTYPE

  # Platform independent types
  cdef enum:
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128

  cdef enum:
    NPY_WRITEABLE, NPY_ALIGNED, NPY_C_CONTIGUOUS, NPY_CONTIGUOUS,
    NPY_FORCECAST, NPY_NOTSWAPPED, NPY_OWNDATA

  # Classes
  ctypedef extern class numpy.dtype [object PyArray_Descr]:
    cdef int type_num, elsize, alignment
    cdef char type, kind, byteorder, hasobject
    cdef object fields, typeobj

  ctypedef extern class numpy.ndarray [object PyArrayObject]:
    cdef char *data
    cdef int nd
    cdef npy_intp *dimensions
    cdef npy_intp *strides
    cdef object base
    cdef dtype descr
    cdef int flags

  ctypedef struct npy_cfloat:
    float real
    float imag

  ctypedef struct npy_cdouble:
    double real
    double imag

  # Functions
  int PyArray_DIM(ndarray arr, int i)
  object PyArray_FROM_OF(object arr, int requirements)

  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)
  dtype PyArray_DescrFromType(int type)
  object PyArray_Scalar(void *data, dtype descr, object base)
  long PyArray_NBYTES(object arr)

  int PyArray_CheckScalar(object sclr)
  void PyArray_ScalarAsCtype(object sclr, void* ptr)
  object PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)
  object PyArray_ContiguousFromAny(object arr, int typenum, int min_depth, int max_depth)
  object PyArray_FROM_OTF(object arr, int typenum, int requirements)

  # The NumPy initialization function
  void import_array()

  void* PyArray_DATA(ndarray arr)





