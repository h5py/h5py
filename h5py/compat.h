/***** Preamble block *********************************************************
* 
* This file is part of h5py, a low-level Python interface to the HDF5 library.
* 
* Copyright (C) 2008 Andrew Collette
* http://h5py.alfven.org
* License: BSD  (See LICENSE.txt for full license)
* 
* $Date$
* 
****** End preamble block ****************************************************/

/* Contains compatibility macros and definitions for use by Cython code */

#ifndef H5PY_COMPAT
#define H5PY_COMPAT

#include <stddef.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "hdf5.h"

/* The HOFFSET macro can't be used from Cython. */

#define h5py_size_n64 (sizeof(npy_complex64))
#define h5py_size_n128 (sizeof(npy_complex128))

#define h5py_offset_n64_real (HOFFSET(npy_complex64, real))
#define h5py_offset_n64_imag (HOFFSET(npy_complex64, imag))
#define h5py_offset_n128_real (HOFFSET(npy_complex128, real))
#define h5py_offset_n128_imag (HOFFSET(npy_complex128, imag))

#endif

